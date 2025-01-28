import torch
import pandas as pd
from tqdm import tqdm
import yaml
from box import Box
import numpy as np
import dataclasses
import os
import logging
from pathlib import Path

from pygpudrive.env.dataset import SceneDataLoader
from eval_utils import load_config, make_env, load_policy, rollout

import pdb

logging.basicConfig(level=logging.INFO)


def evaluate_policy(
    env,
    policy,
    data_loader,
    dataset_name,
    device="cuda",
    deterministic=False,
    render_sim_state=False,
):
    """Evaluate policy in the environment."""

    res_dict = {
        "scene": [],
        "goal_achieved": [],
        "collided": [],
        "off_road": [],
        "not_goal_nor_crashed": [],
        "controlled_agents_in_scene": [],
    }

    for batch in tqdm(
        data_loader,
        desc=f"Processing {dataset_name} batches",
        total=len(data_loader),
        colour="blue",
    ):

        # Update simulator with the new batch of data
        env.swap_data_batch(batch)

        # Rollout policy in the environments
        (
            goal_achieved,
            collided,
            off_road,
            controlled_agents_in_scene,
            not_goal_nor_crashed,
            _,
            _,
        ) = rollout(
            env=env,
            policy=policy,
            device=device,
            deterministic=deterministic,
            render_sim_state=render_sim_state,
        )

        # Get names from env
        scenario_to_worlds_dict = env.get_env_filenames()

        res_dict["scene"].extend(scenario_to_worlds_dict.values())
        res_dict["goal_achieved"].extend(goal_achieved.cpu().numpy())
        res_dict["collided"].extend(collided.cpu().numpy())
        res_dict["off_road"].extend(off_road.cpu().numpy())
        res_dict["not_goal_nor_crashed"].extend(
            not_goal_nor_crashed.cpu().numpy()
        )
        res_dict["controlled_agents_in_scene"].extend(
            controlled_agents_in_scene.cpu().numpy()
        )

    # Convert to pandas dataframe
    df_res = pd.DataFrame(res_dict)
    df_res["dataset"] = dataset_name

    return df_res


if __name__ == "__main__":

    # Load configurations
    eval_config = load_config("examples/experiments/eval/config/eval_config")
    model_config = load_config("examples/experiments/eval/config/model_config")

    train_loader = SceneDataLoader(
        root=eval_config.train_dir,
        batch_size=eval_config.num_worlds,
        dataset_size=eval_config.num_worlds,
        sample_with_replacement=False,
    )

    # Make environment
    env = make_env(eval_config, train_loader)

    for model in model_config.models:

        logging.info(f"Evaluating model {model.name}")

        # Load policy
        policy = load_policy(
            path_to_cpt=model_config.models_path,
            model_name=model.name,
            device=eval_config.device,
            env=env,
        )

        # Create dataloaders for train and test sets
        train_loader = SceneDataLoader(
            root=eval_config.train_dir,
            batch_size=eval_config.num_worlds,
            dataset_size=model.train_dataset_size
            if model.name != "random_baseline"
            else 1000,
            sample_with_replacement=False,
            shuffle=False,
        )

        test_loader = SceneDataLoader(
            root=eval_config.test_dir,
            batch_size=eval_config.num_worlds,
            dataset_size=eval_config.test_dataset_size
            if model.name != "random_baseline"
            else 1000,
            sample_with_replacement=False,
            shuffle=True,
        )

        # Rollouts
        logging.info(
            f"Rollouts on {len(set(train_loader.dataset))} train scenes / {len(set(test_loader.dataset))} test scenes"
        )

        df_res_train = evaluate_policy(
            env=env,
            policy=policy,
            data_loader=train_loader,
            dataset_name="train",
            deterministic=False,
            render_sim_state=False,
        )

        df_res_test = evaluate_policy(
            env=env,
            policy=policy,
            data_loader=test_loader,
            dataset_name="test",
            deterministic=False,
            render_sim_state=False,
        )

        # Concatenate train/test results
        df_res = pd.concat([df_res_train, df_res_test])

        # Add metadata
        df_res["model_name"] = model.name
        df_res["train_dataset_size"] = model.train_dataset_size

        # Store
        if not os.path.exists(eval_config.res_path):
            os.makedirs(eval_config.res_path)

        df_res.to_csv(f"{eval_config.res_path}/{model.name}.csv", index=False)

        logging.info(f"Saved at {eval_config.res_path}/{model.name}.csv \n")
