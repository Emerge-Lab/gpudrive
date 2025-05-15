import torch
import dataclasses
import os
import sys
import mediapy
import logging
import numpy as np
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from box import Box
import yaml
from PIL import Image

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.metadata import Metadata
from gpudrive.datatypes.info import Info
from gpudrive.utils.checkpoint import load_agent
from gpudrive.visualize.utils import img_from_fig
import madrona_gpudrive

# Env Settings
MAX_AGENTS = (
    madrona_gpudrive.kMaxAgentCount
)  # TODO: Set to 128 for real eval
NUM_ENVS = 1
DEVICE = "cuda"  # where to run the env rollouts
INIT_STEPS = 10
DATASET_SIZE = 5
RENDER = True
LOG_DIR = "examples/eval/figures_data/wosac/"
GUIDANCE_MODE = (
    "log_replay"  # Options: "vbd_amortized", "vbd_online", "log_replay"
)
GUIDANCE_DROPOUT_MODE = "avg"  # Options: "max", "avg", "remove_all"
GUIDANCE_DROPOUT_PROB_RANGE = np.arange(0.0, 1.1, 0.1)
SMOOTHEN_TRAJECTORY = True

DATA_PATH = "data/processed/wosac/validation_interactive/json"

CPT_PATH = "checkpoints/model_guidance_logs__R_10000__05_14_16_54_46_975_002500.pt"

# Load agent
agent = load_agent(path_to_cpt=CPT_PATH).to(DEVICE)

config = agent.config

# Save Trajectories:
all_trajectories = []

# For each guidance density, create the env, rollout and collect agent trajectories
for GUIDANCE_DROPOUT_PROB in GUIDANCE_DROPOUT_PROB_RANGE:
    trajectories = []
    # Create data loader
    val_loader = SceneDataLoader(
        root=DATA_PATH,
        batch_size=NUM_ENVS,
        dataset_size=DATASET_SIZE,
        sample_with_replacement=False,
        shuffle=True,
        file_prefix="",
        seed=10,
    )

    # Override default environment settings to match those the agent was trained with
    # TODO(dc): Clean this up
    env_config = EnvConfig(
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        guidance_speed_weight=config.guidance_speed_weight,
        guidance_heading_weight=config.guidance_heading_weight,
        smoothness_weight=config.smoothness_weight,
        norm_obs=config.norm_obs,
        add_previous_action=config.add_previous_action,
        guidance=config.guidance,
        add_reference_pos_xy=config.add_reference_pos_xy,
        add_reference_speed=config.add_reference_speed,
        add_reference_heading=config.add_reference_heading,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        goal_behavior=config.goal_behavior,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=False,
        obs_radius=config.obs_radius,
        max_steer_angle=config.max_steer_angle,
        max_accel_value=config.max_accel_value,
        action_space_steer_disc=config.action_space_steer_disc,
        action_space_accel_disc=config.action_space_accel_disc,
        # Override action space
        steer_actions=torch.round(
            torch.linspace(
                -config.max_steer_angle,
                config.max_steer_angle,
                config.action_space_steer_disc,
            ),
            decimals=3,
        ),
        accel_actions=torch.round(
            torch.linspace(
                -config.max_accel_value,
                config.max_accel_value,
                config.action_space_accel_disc,
            ),
            decimals=3,
        ),
        init_mode="wosac_eval",
        init_steps=INIT_STEPS,
        guidance_mode=GUIDANCE_MODE,
        guidance_dropout_prob=GUIDANCE_DROPOUT_PROB,
        guidance_dropout_mode=GUIDANCE_DROPOUT_MODE,
        smoothen_trajectory=SMOOTHEN_TRAJECTORY,
    )

    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=val_loader,
        max_cont_agents=MAX_AGENTS,
        device=DEVICE,
    )

    # Zero out actions for parked vehicles
    info = Info.from_tensor(
        env.sim.info_tensor(),
        backend=env.backend,
        device=env.device,
    )

    zero_action_mask = (info.off_road == 1) | (
        info.collided_with_vehicle == 1
    ) & (info.type == int(madrona_gpudrive.EntityType.Vehicle))

    control_mask = env.cont_agent_mask.clone().cpu()

    next_obs = env.reset(mask=control_mask)

    # Guidance logging
    num_guidance_points = env.valid_guidance_points
    guidance_densities = num_guidance_points / env.reference_traj_len
    print(
        f"Avg guidance points per agent: {num_guidance_points.cpu().numpy().mean():.2f} which is {guidance_densities.mean().item()*100:.2f} % of the trajectory length (mode = {env.config.guidance_dropout_mode}) \n"
    )

    pos_xy = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend=env.backend,
            device="cpu",
        ).pos_xy[control_mask]

    trajectories.append(pos_xy)

    done_list = [env.get_dones()]

    for time_step in range(env.episode_len - env.init_steps):

        # Predict actions
        action, _, _, _ = agent(next_obs)

        action_template = torch.zeros(
            (env.num_worlds, madrona_gpudrive.kMaxAgentCount), dtype=torch.int64, device=env.device
        )
        action_template[control_mask] = action.to(env.device)

        # Find the integer key for the "do nothing" action (zero steering, zero acceleration)
        # Check using env.action_key_to_values[DO_NOTHING_ACTION_INT]
        DO_NOTHING_ACTION_INT = [
            key
            for key, value in env.action_key_to_values.items()
            if abs(value[0]) == 0.0
            and abs(value[1]) == 0.0
            and abs(value[2]) == 0.0
        ][0]
        action_template[zero_action_mask] = DO_NOTHING_ACTION_INT

        # Step
        env.step_dynamics(action_template)

        # Get next observation
        next_obs = env.get_obs(control_mask)

        # Save to trajectories
        pos_xy = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend=env.backend,
            device="cpu",
        ).pos_xy[control_mask]
        trajectories.append(pos_xy)

        # NOTE(dc): Make sure to decouple the obs from the reward function
        reward = env.get_rewards()
        done = env.get_dones()
        done_list.append(done)
    
    _ = done_list.pop()

    trajectories = torch.stack(trajectories, dim=0).cpu().permute(1, 0, 2)
    all_trajectories.append(trajectories)

all_trajectories = torch.stack(all_trajectories, dim=0).cpu()
all_trajectories = all_trajectories.unsqueeze(0)

# Plot trajectories and save
_ = env.reset(mask=control_mask)

fig = env.vis.plot_simulator_state(
    env_indices=[0],
    agent_positions=all_trajectories,
    zoom_radius=70,
    multiple_rollouts=True,
    line_alpha=0.5,
    line_width=1.0,
    weights=GUIDANCE_DROPOUT_PROB_RANGE,
    colorbar=True,
)[0]

img = Image.fromarray(img_from_fig(fig))
img.save('guidance_density.png')