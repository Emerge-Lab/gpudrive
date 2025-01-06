import torch
import pandas as pd
from tqdm import tqdm
import yaml
from box import Box
import numpy as np
import dataclasses
import os
from pathlib import Path
import mediapy

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.dataset import SceneDataLoader
from pygpudrive.visualize.utils import img_from_fig

from networks.late_fusion import LateFusionTransformer

import logging
logging.basicConfig(level=logging.DEBUG)


def load_policy(path_to_cpt, model_name, device):
    """Load a policy from a given path."""

    # Load the saved checkpoint
    saved_cpt = torch.load(
        f"{path_to_cpt}/{model_name}.pt", map_location=device
    )

    # Create policy architecture from saved checkpoint
    policy = LateFusionTransformer(
        input_dim=saved_cpt["model_arch"]["input_dim"],
        action_dim=saved_cpt["action_dim"],
        hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        pred_heads_arch=saved_cpt["model_arch"]["pred_heads_arch"],
        num_transformer_layers=saved_cpt["model_arch"][
            "num_transformer_layers"
        ],
    ).to(device)

    # Load the model parameters
    policy.load_state_dict(saved_cpt["parameters"])

    return policy.eval()


def rollout(env, policy, device, deterministic=False, render_sim_state=False):
    """Rollout policy in the environment."""

    sim_state_frames = {env_id: [] for env_id in range(env.num_worlds)}

    # Storage
    goal_achieved = torch.zeros(env.num_worlds).to(device)
    collided = torch.zeros(env.num_worlds).to(device)
    off_road = torch.zeros(env.num_worlds).to(device)

    active_worlds = list(range(env.num_worlds))

    next_obs = env.reset()

    # Initialize masks
    live_agent_mask = env.cont_agent_mask.clone()

    for time_step in range(env.config.episode_len):
        
        logging.debug(f"Time step: {time_step}")

        # Get actions
        action, _, _, _ = policy(next_obs[live_agent_mask], deterministic=deterministic)

        # Insert actions at the right positions
        action_template = torch.zeros(
            (env.num_worlds, env.max_agent_count), dtype=torch.int64
        ).to(device)
        action_template[live_agent_mask] = action

        # Step the environment
        env.step_dynamics(action_template)

        if render_sim_state:
            print(f"Rendering time step: {time_step}")

            # Render worlds that are not done
            sim_state_figures = env.vis.plot_simulator_state(
                env_indices=active_worlds,
                time_steps=[time_step] * len(active_worlds),
                zoom_radius=150,
            )
            for idx, env_id in enumerate(active_worlds):
                sim_state_frames[env_id].append(img_from_fig(sim_state_figures[idx]))
                
        # Get infos from the environment
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()

        # Update mask
        live_agent_mask[dones] = False

        # Check terminal envs and store stats
        num_dones_per_world = (dones * env.cont_agent_mask).sum(dim=1)
        total_controlled_agents_per_world = env.cont_agent_mask.sum(dim=1)
        done_worlds = torch.where(
            (num_dones_per_world == total_controlled_agents_per_world)
        )[0]

        if len(done_worlds) > 0:
            for world in done_worlds:
                # If world is not done yet, store scene stats
                if world in active_worlds:
                    active_worlds.remove(world)
                    
                    logging.debug(f"done_worlds: {world} at t = {time_step}")
                    logging.debug(f"goal_achieved: {infos.goal_achieved[world, :]}")

                    goal_achieved[world] = (
                        (
                            infos.goal_achieved[world, :][
                                env.cont_agent_mask[world]
                            ]
                        )
                        .sum()
                        .item()
                    )
                    collided[world] = (
                        infos.collided[world, :][env.cont_agent_mask[world]]
                        .sum()
                        .item()
                    )
                    off_road[world] = (
                        infos.off_road[world, :][env.cont_agent_mask[world]]
                        .sum()
                        .item()
                    )
                
                else:
                    continue

        if dones.all():
            break

    # Aggregate metrics to get average ratio per scene
    goal_achieved = goal_achieved / env.cont_agent_mask.sum(dim=1)
    collided = collided / env.cont_agent_mask.sum(dim=1)
    off_road = off_road / env.cont_agent_mask.sum(dim=1)
    controlled_agents_in_world = env.cont_agent_mask.sum(dim=1)

    return (
        goal_achieved,
        collided,
        off_road,
        controlled_agents_in_world,
        sim_state_frames,
    )


def evaluate_policy(
    env,
    policy,
    data_loader,
    dataset_name,
    device="cuda",
):
    """Evaluate policy in the environment."""

    res_dict = {
        "scene": [],
        "goal_achieved": [],
        "collided": [],
        "off_road": [],
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
            controlled_agents_in_world,
            _,
        ) = rollout(env, policy, device, deterministic=False)

        # Store results for the current batch
        scenario_names = [Path(path).stem for path in batch]
        res_dict["scene"].extend(scenario_names)
        res_dict["goal_achieved"].extend(goal_achieved.cpu().numpy())
        res_dict["collided"].extend(collided.cpu().numpy())
        res_dict["off_road"].extend(off_road.cpu().numpy())
        res_dict["controlled_agents_in_scene"].extend(
            controlled_agents_in_world.cpu().numpy()
        )

    # Convert to pandas dataframe
    df_res = pd.DataFrame(res_dict)
    df_res["dataset"] = dataset_name

    return df_res


def load_config(cfg: str) -> Box:
    """Load configurations as a Box object.
    Args:
        cfg (str): Name of config file.

    Returns:
        Box: Box representation of configurations.
    """
    with open(f"{cfg}.yaml", "r") as stream:
        config = Box(yaml.safe_load(stream))
    return config


def make_env(config, train_loader):
    """Make the environment with the given config."""

    # Override any default environment settings
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        norm_obs=config.normalize_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        dist_to_goal_threshold=config.dist_to_goal_threshold,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=config.use_lidar_obs,
        disable_classic_obs=True if config.use_lidar_obs else False,
        obs_radius=config.obs_radius,
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=config.max_controlled_agents,
        device=config.device,
    )

    return env


if __name__ == "__main__":

    # Load configurations
    setting_config = load_config(
        "examples/experiments/eval/config/setting_config"
    )
    model_config = load_config("examples/experiments/eval/config/model_config")
    
    train_loader = SceneDataLoader(
        root="data/processed/training",
        batch_size=setting_config.num_worlds,
        dataset_size=1000,
        sample_with_replacement=False,
    )

    # Make environment
    env = make_env(setting_config, train_loader)

    for model in model_config.models:

        print(f"Evaluating model: {model.name}")

        # Load policy
        policy = load_policy(
            path_to_cpt=model_config.models_path,
            model_name=model.name,
            device=setting_config.device,
        )

        # Create data loaders
        train_loader = SceneDataLoader(
            root="data/processed/training",
            batch_size=setting_config.num_worlds,
            dataset_size=model.train_dataset_size,
            sample_with_replacement=False,
            shuffle=False, # Don't shuffle because we're using the first N scenes that were also used for training  
        )

        test_loader = SceneDataLoader(
            root="data/processed/testing",
            batch_size=setting_config.num_worlds,
            dataset_size=setting_config.test_dataset_size,
            sample_with_replacement=False,
            shuffle=True,
        )

        # Do rollouts
        df_res_train = evaluate_policy(
            env=env,
            policy=policy,
            data_loader=train_loader,
            dataset_name="train",
        )

        df_res_test = evaluate_policy(
            env=env,
            policy=policy,
            data_loader=test_loader,
            dataset_name="test",
        )

        # Concatenate train/test results
        df_res = pd.concat([df_res_train, df_res_test])

        # Add metadata
        df_res["train_dataset_size"] = model.train_dataset_size

        # Store
        df_res.to_csv(f"{setting_config.res_path}/{model.name}.csv")
        
        print(f'Saved at {setting_config.res_path}/{model.name}.csv')