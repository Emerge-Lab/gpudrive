import torch
import pandas as pd
from tqdm import tqdm
import yaml
from box import Box
import numpy as np
import dataclasses
import os
from pathlib import Path

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.dataset import SceneDataLoader
from pygpudrive.visualize.utils import img_from_fig

from networks.late_fusion import LateFusionTransformer

import logging
import torch
import random

logging.basicConfig(level=logging.INFO)


class RandomPolicy:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n

    def __call__(self, obs, deterministic=False):
        """
        Generate random actions
        """
        # Uniformly sample integers from the action space for each observation
        batch_size = obs.shape[0]
        random_action = torch.randint(
            0, self.action_space_n, (batch_size,), dtype=torch.int64
        )
        return random_action, None, None, None

def load_policy(path_to_cpt, model_name, device):
    """Load a policy from a given path."""

    # Load the saved checkpoint
    if model_name == "random_baseline":
        return RandomPolicy(env.action_space.n)
    
    else: # Load a trained model
        saved_cpt = torch.load(
            f=f"{path_to_cpt}/{model_name}.pt",
            map_location=device,
            weights_only=False,
        )

        logging.info(f"Load model from {path_to_cpt}/{model_name}.pt")

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

        logging.info("Load model parameters")

        return policy.eval()


def rollout(
    env, policy, device, deterministic: bool = False, render_sim_state: bool = False
):
    """
    Perform a rollout of a policy in the environment.

    Args:
        env: The simulation environment.
        policy: The policy to be rolled out.
        device: The device to execute computations on (CPU/GPU).
        deterministic (bool): Whether to use deterministic policy actions.
        render_sim_state (bool): Whether to render the simulation state.

    Returns:
        tuple: Averages for goal achieved, collisions, off-road occurrences, 
               controlled agents count, and simulation state frames.
    """
    # Initialize storage
    sim_state_frames = {env_id: [] for env_id in range(env.num_worlds)}
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len

    # Metrics storage
    goal_achieved = torch.zeros(num_worlds, device=device)
    collided = torch.zeros(num_worlds, device=device)
    off_road = torch.zeros(num_worlds, device=device)

    active_worlds = set(range(num_worlds))
    next_obs = env.reset()
    live_agent_mask = env.cont_agent_mask.clone()

    for time_step in range(episode_len):
        logging.debug(f"Time step: {time_step}")

        # Get actions for active agents
        if live_agent_mask.any():
            action, _, _, _ = policy(
                next_obs[live_agent_mask], deterministic=deterministic
            )

            # Insert actions into a template
            action_template = torch.zeros(
                (num_worlds, max_agent_count), dtype=torch.int64, device=device
            )
            action_template[live_agent_mask] = action.to(device)

            # Step the environment
            env.step_dynamics(action_template)

            if render_sim_state:
                logging.debug(f"Rendering time step: {time_step}")
                sim_state_figures = env.vis.plot_simulator_state(
                    env_indices=list(active_worlds),
                    time_steps=[time_step] * len(active_worlds),
                    zoom_radius=150,
                )
                for idx, env_id in enumerate(active_worlds):
                    sim_state_frames[env_id].append(img_from_fig(sim_state_figures[idx]))

        # Update observations, dones, and infos
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()

        # Update live agent mask
        live_agent_mask[dones] = False

        # Process completed worlds
        num_dones_per_world = (dones & env.cont_agent_mask).sum(dim=1)
        total_controlled_agents = env.cont_agent_mask.sum(dim=1)
        done_worlds = (num_dones_per_world == total_controlled_agents).nonzero(as_tuple=True)[0]

        for world in done_worlds:
            if world in active_worlds:
                active_worlds.remove(world)
                logging.debug(f"World {world} completed at time step {time_step}")

                mask = env.cont_agent_mask[world]
                goal_achieved[world] = infos.goal_achieved[world, mask].sum().item()
                collided[world] = infos.collided[world, mask].sum().item()
                off_road[world] = infos.off_road[world, mask].sum().item()

        if not active_worlds:  # Exit early if all worlds are done
            break

    # Normalize metrics
    controlled_agents_per_world = env.cont_agent_mask.sum(dim=1).float()
    goal_achieved = torch.div(
        goal_achieved, controlled_agents_per_world, rounding_mode='floor'
    ).nan_to_num(0.0)
    collided = torch.div(collided, controlled_agents_per_world, rounding_mode='floor').nan_to_num(0.0)
    off_road = torch.div(off_road, controlled_agents_per_world, rounding_mode='floor').nan_to_num(0.0)

    return (
        goal_achieved,
        collided,
        off_road,
        controlled_agents_per_world,
        sim_state_frames,
    )


def evaluate_policy(
    env,
    policy,
    data_loader,
    dataset_name,
    device="cuda",
    deterministic=False,
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
        ) = rollout(env, policy, device, deterministic=deterministic)

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
        root=setting_config.train_dir,
        batch_size=setting_config.num_worlds,
        dataset_size=100,
        sample_with_replacement=False,
    )

    # Make environment
    env = make_env(setting_config, train_loader)

    for model in model_config.models:

        logging.info(f"Evaluating model {model.name} \n")

        # Load policy
        policy = load_policy(
            path_to_cpt=model_config.models_path,
            model_name=model.name,
            device=setting_config.device,
        )

        # Random policy as baseline
        rand_policy = RandomPolicy(env.action_space.n)

        # Create data loaders
        train_loader = SceneDataLoader(
            root=setting_config.train_dir,
            batch_size=setting_config.num_worlds,
            dataset_size=model.train_dataset_size,
            sample_with_replacement=False,
            shuffle=False,  # Don't shuffle because we're using the first N scenes that were also used for training
        )

        test_loader = SceneDataLoader(
            root=setting_config.test_dir,
            batch_size=setting_config.num_worlds,
            dataset_size=setting_config.test_dataset_size,
            sample_with_replacement=False,
            shuffle=True, 
        )

        logging.info(f'Rollouts on {len(set(train_loader.dataset))} train scenes / {len(set(test_loader.dataset))} test scenes')
        
        # Rollouts
        df_res_train = evaluate_policy(
            env=env,
            policy=policy,
            data_loader=train_loader,
            dataset_name="train",
            deterministic=False,
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
        df_res["model_name"] = model.name
        df_res["train_dataset_size"] = model.train_dataset_size

        # Store
        if not os.path.exists(setting_config.res_path):
            os.makedirs(setting_config.res_path)

        df_res.to_csv(f"{setting_config.res_path}/{model.name}.csv")

        logging.info(f"Saved at {setting_config.res_path}/{model.name}.csv \n")
