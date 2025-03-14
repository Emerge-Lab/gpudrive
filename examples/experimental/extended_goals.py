import os
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from eval_utils import load_policy, rollout, load_config, make_env
from pygpudrive.env.dataset import SceneDataLoader

def visualize_extended_goals(
    env,
    data_loader,
    config,
    save_path,
    num_scenes=None,
):
    """
    Visualize scenes with extended goals for controlled agents.
    
    Args:
        env: Environment instance
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

        # Reset environment to start state
        _ = env.reset()

        # Visualize state with extended goals
        extended_states = env.vis.plot_simulator_state(
            env_indices=list(range(len(batch))),
            time_steps=[0] * len(batch),
            zoom_radius=150,
            extend_goals=True
        )

        # Save extended goal states
        for i, fig in enumerate(extended_states):
            scene_name = Path(batch[i]).stem
            print(f"Scene: {scene_name}\n")
            fig.savefig(os.path.join(save_path, f"{scene_name}_extended.png"))

        scene_count += len(batch)

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
    env = make_env(config, data_loader)

    # Run visualization
    visualize_extended_goals(
        env=env,
        data_loader=data_loader,
        config=config,
        save_path=config.save_results_path,
    )

    logging.info(f"Saved visualizations at {config.save_results_path}")