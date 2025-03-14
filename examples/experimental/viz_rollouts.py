import os
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import mediapy
import numpy as np

from eval_utils import load_policy, rollout, load_config, make_env
from gpudrive.env.dataset import SceneDataLoader

def visualize_rollouts(
    env,
    policy,
    data_loader,
    config,
    save_path,
    num_scenes=None,
    make_videos=True,
):
    """
    Visualize policy rollouts for specified number of scenes.
    
    Args:
        env: Environment instance
        policy: Loaded policy model
        data_loader: SceneDataLoader instance
        config: Configuration object
        save_path: Path to save visualizations
        num_scenes: Number of scenes to visualize. If None, processes all scenes.
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Limit number of scenes if specified
    if num_scenes is not None:
        total_batches = min(num_scenes // config.num_worlds + 1, len(data_loader))
    else:
        total_batches = len(data_loader)
    
    scene_count = 0
    
    for batch_idx, batch in tqdm(
        enumerate(data_loader),
        desc="Processing scenes",
        total=total_batches,
        colour="green",
    ):
        if num_scenes is not None and scene_count >= num_scenes:
            break
            
        # Set new data batch with simulator
        env.swap_data_batch(batch)
        # Rollout policy and get agent positions
        (
            _, _,  # goal_achieved
            _, _,  # collided
            _, _,  # off_road
            _, _,  # not_goal_nor_crashed            
            _,# controlled_agents_in_scene
            sim_state_frames, # sim state frames
            agent_positions,
            _, # episode lengths
        ) = rollout(
            env=env,
            policy=policy,
            device=config.device,
            deterministic=config.deterministic,
            render_sim_state=config.render_sim_state if make_videos else False,
            return_agent_positions=True,
            zoom_radius=40
        )
        
        # Reset environment to visualize final states with trajectories
        _ = env.reset(env.cont_agent_mask)
        final_states = env.vis.plot_simulator_state(
            env_indices=list(range(len(batch))),
            time_steps=[-1] * len(batch),
            zoom_radius = 40,
            agent_positions=agent_positions
        )
        
        # Save final states with trajectories
        for i, fig in enumerate(final_states):
            scene_name = Path(batch[i]).stem
            fig.savefig(os.path.join(save_path, f"{scene_name}_3d.png"))

        if make_videos:
            # Save videos
            sim_state_arrays = {k: np.array(v) for k, v in sim_state_frames.items()}
            filenames = env.get_env_filenames()
            for env_id, frames in sim_state_arrays.items():
            
                filename = filenames[env_id]

                mediapy.write_video(
                    os.path.join(save_path, f"{filename}_3d.gif"),
                    frames,
                    fps=15,
                    codec='gif',
                )

if __name__ == "__main__":
    # Load configuration
    config = load_config("examples/experimental/config/visualization_config")
    
    # Initialize data loader
    data_loader = SceneDataLoader(
        root=config.train_path,
        batch_size=config.num_worlds,
        dataset_size=config.dataset_size,
        sample_with_replacement=False,
    )
    
    # Create environment
    env = make_env(config, data_loader, render_3d=True)
    
    # Load policy
    policy = load_policy(
        path_to_cpt=config.cpt_path,
        model_name=config.cpt_name,
        device=config.device,
        env=env,
    )

    # Run visualization
    visualize_rollouts(
        env=env,
        policy=policy,
        data_loader=data_loader,
        config=config,
        save_path=config.save_results_path,
        num_scenes=None,  # Set to None to process all scenes
    )
    
    logging.info(f"Saved visualizations at {config.save_results_path}")