import torch
import argparse
import os
import numpy as np
import mediapy as media
from PIL import Image
from tqdm import tqdm

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.info import Info
from gpudrive.utils.checkpoint import load_agent
from gpudrive.visualize.utils import img_from_fig
import madrona_gpudrive


def parse_args():
    parser = argparse.ArgumentParser(description='Generate GIFs of scenes with specified guidance density')
    parser.add_argument('--guidance_dropout_prob', type=float, default=0.0,
                        help='Guidance dropout probability (0.0 = full guidance, 1.0 = no guidance)')
    parser.add_argument('--dataset_size', type=int, default=10,
                        help='Number of scenes to process')
    parser.add_argument('--data_path', type=str, 
                        default="data/processed/wosac/validation_interactive/json",
                        help='Path to dataset')
    parser.add_argument('--checkpoint_path', type=str,
                        default="checkpoints/model_guidance_logs__R_10000__05_31_15_21_48_144_014500.pt",
                        help='Path to agent checkpoint')
    parser.add_argument('--output_dir', type=str, default='guidance_gifs',
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


def rollout_scene(env, agent, args):
    """Rollout a single scene and collect frames"""
    
    control_mask = env.cont_agent_mask.clone().cpu()
    next_obs = env.reset(mask=control_mask)
    
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
    
    # Get initial frame
    if 0 % args.render_frequency == 0:
        fig = env.vis.plot_simulator_state(
            env_indices=[0],
            zoom_radius=args.zoom_radius,
            plot_guidance_pos_xy=True,
            center_agent_indices=[0],
        )[0]
        
        # Add guidance info to the plot
        ax = fig.get_axes()[0]
        ax.text(
            0.05, 0.90,
            f"Guidance: {guidance_percent:.1f}%\nDropout prob: {args.guidance_dropout_prob:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            ha="left", va="top",
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=5)
        )
        
        frames.append(img_from_fig(fig))
    
    # Rollout the episode
    for time_step in tqdm(range(env.episode_len - env.config.init_steps), 
                         desc="  Rolling out", leave=False):
        
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
        
        # Render frame if needed
        if (time_step + 1) % args.render_frequency == 0:
            fig = env.vis.plot_simulator_state(
                env_indices=[0],
                zoom_radius=args.zoom_radius,
                plot_guidance_pos_xy=True,
                center_agent_indices=[0],
            )[0]
            
            # Add guidance info to the plot
            ax = fig.get_axes()[0]
            ax.text(
                0.05, 0.90,
                f"Guidance: {guidance_percent:.1f}%\nDropout prob: {args.guidance_dropout_prob:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                color="white",
                ha="left", va="top",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=5)
            )
            
            frames.append(img_from_fig(fig))
    
    return frames, guidance_percent


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating GIFs with guidance dropout probability: {args.guidance_dropout_prob}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup environment and agent
    env, agent = setup_environment(args)
    
    scene_count = 0
    successful_scenes = 0
    
    while scene_count < args.dataset_size:
        try:
            print(f"\nProcessing scene {scene_count + 1}/{args.dataset_size}...")
            
            # Rollout scene and collect frames
            frames, guidance_percent = rollout_scene(env, agent, args)
            
            if frames:
                # Create filename with guidance info
                guidance_str = f"guidance_{guidance_percent:.1f}pct"
                dropout_str = f"dropout_{args.guidance_dropout_prob:.2f}"
                filename = f"scene_{scene_count:03d}_{guidance_str}_{dropout_str}.gif"
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
    
    print(f"\nCompleted! Generated {successful_scenes} GIFs out of {scene_count} scenes.")
    print(f"GIFs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()