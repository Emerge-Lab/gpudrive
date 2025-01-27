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
from pygpudrive.datatypes.observation import GlobalEgoState

from networks.late_fusion import NeuralNet

import logging
import torch
import random

logging.basicConfig(level=logging.INFO)


import pdb


class RandomPolicy:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n

    def __call__(self, obs, deterministic=False):
        """Generate random actions."""
        # Uniformly sample integers from the action space for each observation
        batch_size = obs.shape[0]
        random_action = torch.randint(
            0, self.action_space_n, (batch_size,), dtype=torch.int64
        )
        return random_action, None, None, None


def load_policy(path_to_cpt, model_name, device, env=None):
    """Load a policy from a given path."""

    # Load the saved checkpoint
    if model_name == "random_baseline":
        return RandomPolicy(env.action_space.n)

    else:  # Load a trained model
        saved_cpt = torch.load(
            f=f"{path_to_cpt}/{model_name}.pt",
            map_location=device,
            weights_only=False,
        )

        logging.info(f"Load model from {path_to_cpt}/{model_name}.pt")

        # Create policy architecture from saved checkpoint
        #TODO: Change depending on the network
        policy = LateFusionTransformer(
            input_dim=saved_cpt["model_arch"]["input_dim"],
            action_dim=saved_cpt["action_dim"],
            hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
            pred_heads_arch=saved_cpt["model_arch"]["pred_heads_arch"],
        ).to(device)

        # Load the model parameters
        policy.load_state_dict(saved_cpt["parameters"])

        logging.info("Load model parameters")

        return policy.eval()

def rollout(
    env,
    policy,
    device,
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 5,
    zoom_radius: int = 100,
    results_df: pd.DataFrame = None,
    return_agent_positions: bool = False
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
    agent_positions = torch.zeros((env.num_worlds, env.max_agent_count, episode_len, 2))

    # Reset episode
    next_obs = env.reset()

    # Storage
    goal_achieved = torch.zeros((num_worlds, max_agent_count), device=device)
    collided = torch.zeros((num_worlds, max_agent_count), device=device)
    off_road = torch.zeros((num_worlds, max_agent_count), device=device)
    active_worlds = np.arange(num_worlds).tolist()

    # Note: Should be done on C++ side, these are bug fixes
    infos = env.get_infos()  # Initialize the bugged_agent_mask
    bugged_agent_mask = torch.zeros_like(env.cont_agent_mask, dtype=torch.bool)

    bugged_agent_mask[env.cont_agent_mask] = torch.logical_or(
        infos.off_road[env.cont_agent_mask],
        infos.collided[env.cont_agent_mask],
    )

    logging.info(
        f"Removed {bugged_agent_mask.sum()} bugged agents; {(bugged_agent_mask.sum()/env.cont_agent_mask.sum())*100:.2f}% of controlled agents \n"
    )

    controlled_agent_mask = env.cont_agent_mask.clone() & ~bugged_agent_mask

    live_agent_mask = controlled_agent_mask.clone()

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

            if render_sim_state and len(active_worlds) > 0:
                has_live_agent = torch.where(
                    live_agent_mask[active_worlds, :].sum(axis=1) > 0
                )[0].tolist()

                if time_step % render_every_n_steps == 0:

                    logging.info(f"Rendering time step {time_step}")

                    sim_state_figures = env.vis.plot_simulator_state(
                        env_indices=has_live_agent,
                        time_steps=[time_step] * len(has_live_agent),
                        zoom_radius=zoom_radius,
                        results_df=results_df,
                        eval_mode=True,
                        agent_positions = agent_positions
                    )
                    for idx, env_id in enumerate(has_live_agent):
                        sim_state_frames[env_id].append(
                            img_from_fig(sim_state_figures[idx])
                        )

        # Update observations, dones, and infos
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()
        
        #pdb.set_trace()
        
        if return_agent_positions:
            global_agent_states = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor())
            agent_positions[:, :, time_step, 0] = global_agent_states.pos_x
            agent_positions[:, :, time_step, 1] = global_agent_states.pos_y            

        # Count the collisions, off-road occurrences, and goal achievements
        # at a given time step for living agents
        off_road[live_agent_mask] += infos.off_road[live_agent_mask]
        collided[live_agent_mask] += infos.collided[live_agent_mask]
        goal_achieved[live_agent_mask] += infos.goal_achieved[live_agent_mask]

        logging.debug(f"active_worlds: {active_worlds}")
        logging.debug(f"num_agents_live: {live_agent_mask.sum()}")

        # Update live agent mask
        live_agent_mask[dones] = False

        # Process completed worlds
        num_dones_per_world = (dones & controlled_agent_mask).sum(dim=1)
        total_controlled_agents = controlled_agent_mask.sum(dim=1)
        done_worlds = (num_dones_per_world == total_controlled_agents).nonzero(
            as_tuple=True
        )[0]

        for world in done_worlds:
            if world in active_worlds:
                active_worlds.remove(world)
                logging.debug(f"World {world} done at time step {time_step}")

        if not active_worlds:  # Exit early if all worlds are done
            break

    # Aggregate metrics to obtain averages across scenes
    controlled_agents_per_scene = controlled_agent_mask.sum(dim=1).float()
    goal_achieved_per_scene = (goal_achieved > 0).float().sum(
        axis=1
    ) / controlled_agents_per_scene
    collided_per_scene = (collided > 0).float().sum(
        axis=1
    ) / controlled_agents_per_scene
    off_road_per_scene = (off_road > 0).float().sum(
        axis=1
    ) / controlled_agents_per_scene
    not_goal_nor_crash = torch.logical_and(
        goal_achieved == 0,  # Didn't reach the goal
        torch.logical_and(
            collided == 0,  # Didn't collide
            torch.logical_and(
                off_road == 0,  # Didn't go off-road
                controlled_agent_mask,  # Only count controlled agents
            ),
        ),
    )
    not_goal_nor_crash_per_scene = (
        not_goal_nor_crash.float().sum(dim=1) / controlled_agents_per_scene
    )

    return (
        goal_achieved_per_scene,
        collided_per_scene,
        off_road_per_scene,
        controlled_agents_per_scene,
        not_goal_nor_crash_per_scene,
        sim_state_frames,
        agent_positions,
    )

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
        norm_obs=config.norm_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        dist_to_goal_threshold=config.dist_to_goal_threshold,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=config.lidar_obs,
        disable_classic_obs=True if config.lidar_obs else False,
        obs_radius=config.obs_radius,
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=config.max_controlled_agents,
        device=config.device,
    )

    return env