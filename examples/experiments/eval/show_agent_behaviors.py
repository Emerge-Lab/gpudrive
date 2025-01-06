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

from evaluate import load_policy, load_config, make_env, rollout
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def make_videos(data_batch, env, policy, device):
    """Make videos policy rollouts environment."""

    # Update simulator with the provided data batch
    env.swap_data_batch(data_batch)

    # Rollout policy in the environments
    _, _, _, _, sim_state_frames = rollout(
        env, policy, device, render_sim_state=True
    )

    return sim_state_frames

if __name__ == "__main__":
    
    # Load configurations
    setting_config = load_config(
        "examples/experiments/eval/config/setting_config"
    )
    model_config = load_config("examples/experiments/eval/config/model_config")
    
    # Data loader
    train_loader = SceneDataLoader(
        root="data/processed/training",
        batch_size=setting_config.num_worlds,
        dataset_size=10,
        sample_with_replacement=False,
    )

    # Make environment
    env = make_env(setting_config, train_loader)
 
    # Load policy
    policy = load_policy(
        path_to_cpt=model_config.models_path,
        model_name=model_config['models'][0].name,
        device=setting_config.device,
    )

    # Show all training vids
    selected_batch_train = env.data_batch

    sim_state_frames_train = make_videos(
        data_batch=selected_batch_train,
        env=env,
        policy=policy,
        device=setting_config.device,
    )

    # Test
    # base_path = "data/processed/testing/"
    # selected_scenes_test = df_res_test.scene.values
    # selected_batch_test = [
    #     f"{base_path}{scene}.json" for scene in selected_scenes_test
    # ][:10]

    # sim_state_frames_test = make_videos(
    #     data_batch=selected_batch_test,
    #     env=env,
    #     policy=policy,
    #     device=setting_config.device,
    # )

    # Convert to set of numpy arrays
    sim_state_arrays_train = {
        k: np.array(v) for k, v in sim_state_frames_train.items()
    }
    # sim_state_arrays_test = {
    #     k: np.array(v) for k, v in sim_state_frames_test.items()
    # }
    import wandb

    # Initialize a W&B run (ensure you have called `wandb.init()` earlier in the script)
    wandb.init(project="self_play_eval", job_type="video_logging")

    # Log videos to W&B
    for env_id, frames in sim_state_arrays_train.items():
        
        # Convert frames (numpy arrays) to video
        # Needs (Frames, Channels, Height, Width) format
        video_array = wandb.Video(np.moveaxis(frames, -1, 1), fps=15, format="mp4")
        
        # Log video to W&B
        wandb.log({f"train_scene_{env_id}": video_array})
        print(f"Logged video")