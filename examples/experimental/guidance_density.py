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
from gpudrive.datatypes.trajectory import to_local_frame
import madrona_gpudrive

# Env Settings
MAX_AGENTS = (
    madrona_gpudrive.kMaxAgentCount
)  # TODO: Set to 128 for real eval
NUM_ENVS = 1
DEVICE = "cuda"  # where to run the env rollouts
INIT_STEPS = 10
DATASET_SIZE = 20
RENDER = True
LOG_DIR = "examples/eval/figures_data/wosac/"
GUIDANCE_MODE = (
    "log_replay"  # Options: "vbd_amortized", "vbd_online", "log_replay"
)
GUIDANCE_DROPOUT_MODE = "avg"  # Options: "max", "avg", "remove_all"
GUIDANCE_DROPOUT_PROB_RANGE = np.arange(0.0, 1.1, 0.33)
SMOOTHEN_TRAJECTORY = True

DATA_PATH = "data/processed/wosac/validation_interactive/json"

CPT_PATH = "checkpoints/model_guidance_logs__R_10000__05_31_15_21_48_144_014500.pt"

# Load agent
agent = load_agent(path_to_cpt=CPT_PATH).to(DEVICE)

config = agent.config

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
    action_space_steer_disc=config.action_space_steer_disc,
    action_space_accel_disc=config.action_space_accel_disc,
    init_mode="wosac_eval",
    init_steps=INIT_STEPS,
    guidance_mode=GUIDANCE_MODE,
    guidance_dropout_prob=GUIDANCE_DROPOUT_PROB_RANGE[0],  # Set to 0 for the first run
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

def transform_trajectories_to_local_frame(global_trajectories, ego_pos, ego_yaw, device):
    """
    Transform trajectories from simulator coordinates to local (ego-centric) coordinates.
    
    Args:
        global_trajectories: Tensor of shape [n_rollouts, n_steps, 2] (x, y positions in simulator coords)
        ego_pos: Tensor of shape [2] (ego x, y position in simulator coords)
        ego_yaw: Scalar tensor (ego heading angle)
        device: Device to run computations on
        
    Returns:
        local_trajectories: Tensor of shape [n_rollouts, n_steps, 2] in ego frame
    """
    n_rollouts, n_steps, _ = global_trajectories.shape
    local_trajectories = torch.zeros_like(global_trajectories)
    
    for rollout_idx in range(n_rollouts):
        for step_idx in range(n_steps):
            global_pos = global_trajectories[rollout_idx, step_idx, :]
            local_pos = to_local_frame(
                global_pos_xy=global_pos.unsqueeze(0),  # Add batch dimension
                ego_pos=ego_pos,
                ego_yaw=ego_yaw,
                device=device,
            )
            local_trajectories[rollout_idx, step_idx, :] = local_pos.squeeze(0)
    
    return local_trajectories

# Create output directory
os.makedirs('guidance_density', exist_ok=True)

scene_count = 0
while scene_count < DATASET_SIZE:

    # Save Trajectories for the first controlled agent in global coordinates
    all_global_trajectories = []
    
    # Get the first controlled agent index for this scene
    control_mask = env.cont_agent_mask.clone().cpu()
    first_controlled_agent_idx = torch.where(control_mask[0])[0][0].item()
    
    # For each guidance density, rollout and collect agent trajectories
    for GUIDANCE_DROPOUT_PROB in GUIDANCE_DROPOUT_PROB_RANGE:
        
        # Update the environment configuration for the current guidance dropout probability
        env.config.guidance_dropout_prob = GUIDANCE_DROPOUT_PROB
        control_mask = env.cont_agent_mask.clone().cpu()
        next_obs = env.reset(mask=control_mask)

        global_trajectories = []

        # Zero out actions for parked vehicles
        info = Info.from_tensor(
            env.sim.info_tensor(),
            backend=env.backend,
            device=env.device,
        )

        zero_action_mask = (info.off_road == 1) | (
            info.collided_with_vehicle == 1
        ) & (info.type == int(madrona_gpudrive.EntityType.Vehicle))

        # Guidance logging
        num_guidance_points = env.valid_guidance_points
        guidance_densities = num_guidance_points / env.reference_traj_len
        print(
            f"Avg guidance points per agent: {num_guidance_points.cpu().numpy().mean():.2f} which is {guidance_densities.mean().item()*100:.2f} % of the trajectory length (mode = {env.config.guidance_dropout_mode}) \n"
        )

        # Get position in simulator coordinates (with world mean already subtracted)
        global_agent_states = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend=env.backend,
            device="cpu",
        )
        
        # Store trajectory for the first controlled agent only (already in simulator coordinates)
        first_agent_pos = global_agent_states.pos_xy[0, first_controlled_agent_idx, :]
        global_trajectories.append(first_agent_pos)

        done_list = [env.get_dones()]

        for time_step in range(env.episode_len - env.init_steps):

            # Predict actions
            action, _, _, _ = agent(next_obs)

            action_template = torch.zeros(
                (env.num_worlds, madrona_gpudrive.kMaxAgentCount), dtype=torch.int64, device=env.device
            )
            action_template[control_mask] = action.to(env.device)

            # Find the integer key for the "do nothing" action (zero steering, zero acceleration)
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

            # Save to trajectories in simulator coordinates (world mean already subtracted)
            global_agent_states = GlobalEgoState.from_tensor(
                env.sim.absolute_self_observation_tensor(),
                backend=env.backend,
                device="cpu",
            )
            
            # Store trajectory for the first controlled agent only (already in simulator coordinates)
            first_agent_pos = global_agent_states.pos_xy[0, first_controlled_agent_idx, :]
            global_trajectories.append(first_agent_pos)

            reward = env.get_rewards()
            done = env.get_dones()
            done_list.append(done)
        
        _ = done_list.pop()

        # Stack trajectories for this rollout: shape [n_steps, 2]
        rollout_trajectory = torch.stack(global_trajectories, dim=0)
        all_global_trajectories.append(rollout_trajectory)

    # Stack all rollouts: shape [n_rollouts, n_steps, 2]
    all_global_trajectories = torch.stack(all_global_trajectories, dim=0)
    
    # Reset environment to get the initial state for egocentric transformation
    control_mask = env.cont_agent_mask.clone().cpu()
    env.config.guidance_dropout_prob = 0.0  # Reset to no dropout for final state
    _ = env.reset(mask=control_mask)
    
    # Get ego agent's initial position and heading for coordinate transformation
    # Use simulator coordinates (world mean already subtracted) - this is what to_local_frame expects
    global_agent_states = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend=env.backend,
        device="cpu",
    )
    
    ego_pos = global_agent_states.pos_xy[0, first_controlled_agent_idx, :]  # [x, y] in simulator coords
    ego_yaw = global_agent_states.rotation_angle[0, first_controlled_agent_idx]  # scalar
    
    # Transform all trajectories to ego-centric coordinates
    local_trajectories = transform_trajectories_to_local_frame(
        all_global_trajectories, ego_pos, ego_yaw, device="cpu"
    )
    
    # Create a combined trajectory array with weights for coloring
    # Shape: [n_rollouts, n_steps, 2]
    trajectory_weights = 1 - GUIDANCE_DROPOUT_PROB_RANGE  # Higher weight = more guidance
    
    # Plot using the egocentric view
    fig = env.vis.plot_agent_observation(
        env_idx=0,
        agent_idx=first_controlled_agent_idx,
        figsize=(12, 12),
        trajectory=None,  # We'll modify the function to handle multiple trajectories
        step_reward=None,
        route_progress=None,
    )
    
    # Manually add the multiple trajectories with different colors based on guidance density
    if fig is not None:
        ax = fig.get_axes()[0]
        
        # Color trajectories based on guidance density
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create colormap for guidance density
        colors = sns.color_palette("viridis", len(GUIDANCE_DROPOUT_PROB_RANGE))
        
        for rollout_idx, (trajectory, dropout_prob) in enumerate(zip(local_trajectories, GUIDANCE_DROPOUT_PROB_RANGE)):
            # Filter out invalid points
            valid_mask = (
                (trajectory[:, 0] != 0) & 
                (trajectory[:, 1] != 0) & 
                (torch.abs(trajectory[:, 0]) < 1000) & 
                (torch.abs(trajectory[:, 1]) < 1000)
            )
            
            if valid_mask.sum() > 1:
                valid_trajectory = trajectory[valid_mask]
                
                # Plot trajectory line
                ax.plot(
                    valid_trajectory[:, 0].cpu().numpy(),
                    valid_trajectory[:, 1].cpu().numpy(),
                    color=colors[rollout_idx],
                    linewidth=2.0,
                    alpha=0.8,
                    label=f'Guidance: {(1-dropout_prob)*100:.0f}%'
                )
                
                # Plot trajectory points
                ax.scatter(
                    valid_trajectory[:, 0].cpu().numpy(),
                    valid_trajectory[:, 1].cpu().numpy(),
                    color=colors[rollout_idx],
                    s=15,
                    alpha=0.6,
                    zorder=10
                )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Add title
        ax.set_title(f'Egocentric View - Scene {scene_count}\nAgent {first_controlled_agent_idx} Trajectories by Guidance Density', 
                    fontsize=14, pad=20)
        
        # Save the figure
        img = Image.fromarray(img_from_fig(fig))
        img.save(f'guidance_density/scene_{scene_count}_agent_{first_controlled_agent_idx}.png')
        
        plt.close(fig)

    print(f"Processed scene {scene_count} with agent {first_controlled_agent_idx}")
    scene_count += 1

    try:
        env.swap_data_batch()
    except StopIteration:
        # If we run out of scenes, break the loop
        print("No more scenes in the dataset.")
        break

print(f"Generated {scene_count} egocentric visualization figures in 'guidance_density/' directory")