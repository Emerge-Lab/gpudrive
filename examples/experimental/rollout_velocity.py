import torch
import argparse
import os
import numpy as np
import mediapy as media
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import LocalEgoState
from gpudrive.datatypes.info import Info
from gpudrive.utils.checkpoint import load_agent
from gpudrive.visualize.utils import img_from_fig
import madrona_gpudrive


def parse_args():
    parser = argparse.ArgumentParser(description='Generate GIFs of scenes with velocity graphs')
    parser.add_argument('--guidance_dropout_prob', type=float, default=0.0,
                        help='Guidance dropout probability (0.0 = full guidance, 1.0 = no guidance)')
    parser.add_argument('--dataset_size', type=int, default=10,
                        help='Number of scenes to process')
    parser.add_argument('--data_path', type=str, 
                        default="data/processed/wosac/validation_interactive/json",
                        help='Path to dataset')
    parser.add_argument('--checkpoint_path', type=str,
                        default="checkpoints/model_guidance_logs__R_10000__06_07_13_55_31_201_013500.pt",
                        help='Path to agent checkpoint')
    parser.add_argument('--output_dir', type=str, default='velocity_gifs',
                        help='Output directory for GIFs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--zoom_radius', type=int, default=45,
                        help='Zoom radius for visualization')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for GIF')
    parser.add_argument('--render_frequency', type=int, default=1,
                        help='Render every N timesteps (1 = every timestep)')
    parser.add_argument('--guidance_mode', type=str, default='log_replay',
                        choices=['log_replay', 'vbd_amortized', 'vbd_online'],
                        help='Guidance mode to use')
    parser.add_argument('--guidance_dropout_mode', type=str, default='avg',
                        choices=['max', 'avg', 'remove_all'],
                        help='Guidance dropout mode')
    parser.add_argument('--ego_agent_idx', type=int, default=0,
                        help='Index of the ego agent to track (default: 0 for first controlled agent)')
    parser.add_argument('--velocity_graph_height', type=float, default=0.3,
                        help='Height ratio of velocity graph relative to total figure (0.0-1.0)')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup the environment and agent"""
    
    # Environment constants
    MAX_AGENTS = madrona_gpudrive.kMaxAgentCount
    NUM_ENVS = 1
    INIT_STEPS = 10
    
    # Load agent
    print(f"Loading agent from {args.checkpoint_path}...")
    agent = load_agent(path_to_cpt=args.checkpoint_path).to(args.device)
    config = agent.config
    
    # Create data loader
    print(f"Loading dataset from {args.data_path}...")
    val_loader = SceneDataLoader(
        root=args.data_path,
        batch_size=NUM_ENVS,
        dataset_size=args.dataset_size,
        sample_with_replacement=False,
        shuffle=True,
        file_prefix="",
        seed=10,
    )
    
    # Override default environment settings
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
        guidance_mode=args.guidance_mode,
        guidance_dropout_prob=args.guidance_dropout_prob,
        guidance_dropout_mode=args.guidance_dropout_mode,
        smoothen_trajectory=True,
    )
    
    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=val_loader,
        max_cont_agents=MAX_AGENTS,
        device=args.device,
    )
    
    return env, agent


def create_combined_figure(scene_fig, velocity_data, current_timestep, total_timesteps, 
                          args, guidance_percent, ego_agent_idx):
    """Create a combined figure with scene and velocity graph"""
    
    # Get the scene image from the existing figure
    scene_img = img_from_fig(scene_fig)
    scene_height, scene_width = scene_img.shape[:2]
    
    # Calculate dimensions
    velocity_height_ratio = args.velocity_graph_height
    velocity_height = int(scene_height * velocity_height_ratio)
    total_height = scene_height + velocity_height
    
    # Create new figure with custom layout
    fig = plt.figure(figsize=(12, 12 * total_height / scene_width))
    
    # Create grid layout: scene on top, velocity graph on bottom
    gs = GridSpec(2, 1, height_ratios=[1-velocity_height_ratio, velocity_height_ratio], 
                  hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Add scene as image in top subplot
    ax_scene = fig.add_subplot(gs[0])
    ax_scene.imshow(scene_img)
    ax_scene.set_xlim(0, scene_width)
    ax_scene.set_ylim(scene_height, 0)  # Flip y-axis for image
    ax_scene.set_aspect('equal')
    ax_scene.axis('off')
    
    # Add velocity graph in bottom subplot
    ax_velocity = fig.add_subplot(gs[1])
    
    # Extract velocity data up to current timestep
    timesteps = velocity_data['timesteps'][:current_timestep + 1]
    velocities = velocity_data['velocities'][:current_timestep + 1]
    
    # Plot velocity history
    if len(timesteps) > 1:
        ax_velocity.plot(timesteps, velocities, 'b-', linewidth=2, alpha=0.7)
        
        # Add current point
        if current_timestep < len(velocity_data['velocities']):
            current_vel = velocity_data['velocities'][current_timestep]
            ax_velocity.plot(timesteps[-1], current_vel, 'ro', markersize=8, 
                           markeredgecolor='black', markeredgewidth=1)
    
    # Set up the graph
    ax_velocity.set_xlim(0, total_timesteps)
    ax_velocity.set_ylim(0, max(max(velocity_data['velocities']) * 1.1, 1.0))
    ax_velocity.set_xlabel('Time Step', fontsize=12)
    ax_velocity.set_ylabel('Velocity (m/s)', fontsize=12)
    ax_velocity.grid(True, alpha=0.3)
    ax_velocity.set_title(f'Ego Agent {ego_agent_idx} Velocity | Guidance: {guidance_percent:.1f}%', 
                         fontsize=14, pad=10)
    
    # Add vertical line at current timestep
    ax_velocity.axvline(x=current_timestep, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    return fig


def rollout_scene(env, agent, args):
    """Rollout a single scene and collect frames with velocity data"""
    
    control_mask = env.cont_agent_mask.clone().cpu()
    next_obs = env.reset(mask=control_mask)
    
    # Get the ego agent index (validate it exists and is controlled)
    controlled_indices = torch.where(control_mask[0])[0]
    
    if args.ego_agent_idx >= len(controlled_indices):
        print(f"  Warning: ego_agent_idx {args.ego_agent_idx} out of range. "
              f"Only {len(controlled_indices)} controlled agents. Using agent 0.")
        ego_agent_idx = controlled_indices[0].item()
    else:
        ego_agent_idx = controlled_indices[args.ego_agent_idx].item()
    
    print(f"  Tracking ego agent index: {ego_agent_idx}")
    
    # Zero out actions for parked vehicles
    info = Info.from_tensor(
        env.sim.info_tensor(),
        backend=env.backend,
        device=env.device,
    )
    
    zero_action_mask = (info.off_road == 1) | (
        info.collided_with_vehicle == 1
    ) & (info.type == int(madrona_gpudrive.EntityType.Vehicle))
    
    # Log guidance info
    num_guidance_points = env.valid_guidance_points
    guidance_densities = num_guidance_points / env.reference_traj_len
    guidance_percent = guidance_densities.mean().item() * 100
    
    print(f"  Guidance density: {guidance_percent:.1f}% "
          f"(avg {num_guidance_points.cpu().numpy().mean():.1f} points)")
    
    frames = []
    velocity_data = {
        'timesteps': [],
        'velocities': [],
    }
    
    total_timesteps = env.episode_len - env.config.init_steps
    
    # Get initial velocity
    local_ego_states = LocalEgoState.from_tensor(
        env.sim.self_observation_tensor(),
        backend=env.backend,
        device="cpu",
    )
    initial_velocity = local_ego_states.speed[0, ego_agent_idx].item()
    velocity_data['timesteps'].append(0)
    velocity_data['velocities'].append(initial_velocity)
    
    # Get initial frame
    if 0 % args.render_frequency == 0:
        scene_fig = env.vis.plot_simulator_state(
            env_indices=[0],
            zoom_radius=args.zoom_radius,
            plot_guidance_pos_xy=True,
            center_agent_indices=[ego_agent_idx],
        )[0]
        
        # Create combined figure with velocity graph
        combined_fig = create_combined_figure(
            scene_fig, velocity_data, 0, total_timesteps, 
            args, guidance_percent, ego_agent_idx
        )
        
        frames.append(img_from_fig(combined_fig))
        plt.close(scene_fig)
        plt.close(combined_fig)
    
    # Rollout the episode
    for time_step in tqdm(range(total_timesteps), desc="  Rolling out", leave=False):
        
        # Predict actions
        action, _, _, _ = agent(next_obs)
        
        action_template = torch.zeros(
            (env.num_worlds, madrona_gpudrive.kMaxAgentCount), 
            dtype=torch.int64, device=env.device
        )
        action_template[control_mask] = action.to(env.device)
        
        # Find the "do nothing" action for parked vehicles
        DO_NOTHING_ACTION_INT = [
            key for key, value in env.action_key_to_values.items()
            if abs(value[0]) == 0.0 and abs(value[1]) == 0.0 and abs(value[2]) == 0.0
        ][0]
        action_template[zero_action_mask] = DO_NOTHING_ACTION_INT
        
        # Step environment
        env.step_dynamics(action_template)
        next_obs = env.get_obs(control_mask)
        
        # Get velocity data for ego agent
        local_ego_states = LocalEgoState.from_tensor(
            env.sim.self_observation_tensor(),
            backend=env.backend,
            device="cpu",
        )
        
        # Get speed directly from LocalEgoState
        velocity = local_ego_states.speed[0, ego_agent_idx].item()
        velocity_data['timesteps'].append(time_step + 1)
        velocity_data['velocities'].append(velocity)
        
        # Render frame if needed
        if (time_step + 1) % args.render_frequency == 0:
            scene_fig = env.vis.plot_simulator_state(
                env_indices=[0],
                zoom_radius=args.zoom_radius,
                plot_guidance_pos_xy=True,
                center_agent_indices=[ego_agent_idx],
            )[0]
            
            # Create combined figure with velocity graph
            combined_fig = create_combined_figure(
                scene_fig, velocity_data, time_step + 1, total_timesteps,
                args, guidance_percent, ego_agent_idx
            )
            
            frames.append(img_from_fig(combined_fig))
            plt.close(scene_fig)
            plt.close(combined_fig)
    
    return frames, guidance_percent, ego_agent_idx, velocity_data


def main():
    args = parse_args()
    
    # Validate arguments
    if not (0.0 <= args.velocity_graph_height <= 1.0):
        print("Error: velocity_graph_height must be between 0.0 and 1.0")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating GIFs with velocity graphs")
    print(f"Guidance dropout probability: {args.guidance_dropout_prob}")
    print(f"Ego agent index: {args.ego_agent_idx}")
    print(f"Velocity graph height ratio: {args.velocity_graph_height}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup environment and agent
    env, agent = setup_environment(args)
    
    scene_count = 0
    successful_scenes = 0
    
    while scene_count < args.dataset_size:
        try:
            print(f"\nProcessing scene {scene_count + 1}/{args.dataset_size}...")
            
            # Rollout scene and collect frames with velocity data
            frames, guidance_percent, actual_ego_idx, velocity_data = rollout_scene(env, agent, args)
            
            if frames:
                # Create filename with guidance and ego agent info
                guidance_str = f"guidance_{guidance_percent:.1f}pct"
                dropout_str = f"dropout_{args.guidance_dropout_prob:.2f}"
                ego_str = f"ego_{actual_ego_idx}"
                filename = f"scene_{scene_count:03d}_{guidance_str}_{dropout_str}_{ego_str}_velocity.gif"
                filepath = os.path.join(args.output_dir, filename)
                
                # Save GIF
                print(f"  Saving GIF with {len(frames)} frames to {filename}")
                media.write_video(
                    filepath, 
                    np.array(frames), 
                    fps=args.fps, 
                    codec="gif"
                )
                
                successful_scenes += 1
            else:
                print(f"  Warning: No frames generated for scene {scene_count}")
            
            scene_count += 1
            
            # Try to load next scene
            try:
                env.swap_data_batch()
            except StopIteration:
                print("No more scenes in the dataset.")
                break
                
        except Exception as e:
            print(f"  Error processing scene {scene_count}: {e}")
            scene_count += 1
            try:
                env.swap_data_batch()
            except StopIteration:
                break
    
    print(f"\nCompleted! Generated {successful_scenes} GIFs with velocity graphs out of {scene_count} scenes.")
    print(f"GIFs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()