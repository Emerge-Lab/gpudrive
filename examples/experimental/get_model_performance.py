import torch
import pandas as pd
from box import Box
import numpy as np
import os
import logging
from gpudrive.env.dataset import SceneDataLoader
from eval_utils import (
    load_config,
    make_env,
    load_policy,
    evaluate_policy,
)

import random
import torch
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)
SEED = 42  # Set to any fixed value
set_seed(SEED)

if __name__ == "__main__":

    # Load configurations
    eval_config = load_config("examples/experimental/config/eval_config")
    model_config = load_config("examples/experimental/config/model_config")

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

        tab_agg_perf = df_res.groupby("dataset")[
            [
                "goal_achieved_frac",
                "collided_frac",
                "off_road_frac",
                "other_frac",
            ]
        ].agg(["mean", "std"])
        tab_agg_perf = tab_agg_perf * 100
        tab_agg_perf = tab_agg_perf.round(1)

        print("Scene-based metrics \n")
        print(tab_agg_perf)
        print("")

        print("Agent-based metrics \n")
        total_agents = df_res["controlled_agents_in_scene"].sum()
        collision_rate = (df_res["collided_count"].sum() / total_agents) * 100
        offroad_rate = (df_res["off_road_count"].sum() / total_agents) * 100
        goal_rate = (df_res["goal_achieved_count"].sum() / total_agents) * 100
        other_rate = (df_res["other_count"].sum() / total_agents) * 100

        print(f"Total agents: {total_agents} in {df_res.shape[0]} scenes")
        print(f"Collision rate: {collision_rate}")
        print(f"Offroad rate: {offroad_rate}")
        print(f"Goal rate: {goal_rate}")
        print(f"Other rate: {other_rate}")

        df_res.to_csv(f"{eval_config.res_path}/{model.name}.csv", index=False)

        logging.info(f"Saved at {eval_config.res_path}/{model.name}.csv \n")
