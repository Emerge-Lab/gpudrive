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
import pdb

logging.basicConfig(level=logging.INFO)

def make_videos(data_batch, env, policy, device, zoom_radius, results_df):
    """Make videos policy rollouts environment."""
    # Update simulator with the provided data batch
    env.swap_data_batch(data_batch)

    # Rollout policy in the environments
    _, _, _, _, _, sim_state_frames = rollout(
        env=env, 
        policy=policy, 
        device=device, 
        deterministic=False,
        render_sim_state=True,
        render_every_n_steps=3,
        zoom_radius=zoom_radius,
        results_df=results_df
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
        root=setting_config.train_dir,
        batch_size=100,
        dataset_size=100,
        sample_with_replacement=False,
        shuffle=False
    )

    # Make environment
    env = make_env(setting_config, train_loader)
    
    MODEL_TO_LOAD = model_config['models'][0].name
 
    # Load policy
    policy = load_policy(
        path_to_cpt=model_config.models_path,
        model_name=MODEL_TO_LOAD,
        device=setting_config.device,
    )
    
    logging.info(f"Loaded policy {MODEL_TO_LOAD}")
    
    # Load results dataframe
    results_df = pd.read_csv("examples/experiments/eval/dataframes/model_PPO__R_100__01_10_17_06_33_696_003000.csv")
    results_df_selected = results_df[results_df['dataset'] == 'train']

    #pdb.set_trace()
    
    # TODO: Select scenes from results df

    # Select data 
    selected_batch_train = env.data_batch

    # Rollout policy and make videos
    sim_state_frames_train = make_videos(
        data_batch=selected_batch_train,
        env=env,
        policy=policy,
        device=setting_config.device,
        zoom_radius=100,
        results_df=results_df_selected
    )

    sim_state_arrays_train = {
        k: np.array(v) for k, v in sim_state_frames_train.items()
    }
    
    videos_dir = Path(f"videos/{MODEL_TO_LOAD}")
    videos_dir.mkdir(exist_ok=True)

    # Save videos locally
    for env_id, frames in sim_state_arrays_train.items():
    
        video_path = videos_dir / f"train_scene_{env_id}.mp4"
        
        mediapy.write_video(
            str(video_path),
            frames,
            fps=5, 
        )
        
        logging.info(f"Saved video to {video_path}")