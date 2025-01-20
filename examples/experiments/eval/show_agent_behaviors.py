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

def make_videos(
    df_results, 
    policy, 
    eval_config,
    sort_by=None,
    show_top_k=100,
    device='cuda', 
    zoom_radius=100, 
    deterministic=False,
    render_every_n_steps=10,
    ):
    """Make videos policy rollouts environment."""
    
    # Make environment
    train_loader = SceneDataLoader(
        root=eval_config.train_dir,
        batch_size=show_top_k,
        dataset_size=show_top_k,
        sample_with_replacement=False,
        shuffle=False
    )
    
    env = make_env(eval_config, train_loader)    
    
    # Sample data batch from dataframe
    if sort_by == 'failures':
        pass
    elif sort_by == 'success':
        pass
    elif sort_by is None:
        data_batch = env.data_batch
    
    # Update simulator with the provided data batch
    env.swap_data_batch(data_batch)

    # Rollout policy in the environments
    _, _, _, _, _, sim_state_frames = rollout(
        env=env, 
        policy=policy, 
        device=device, 
        deterministic=deterministic,
        render_sim_state=True,
        render_every_n_steps=render_every_n_steps,
        zoom_radius=zoom_radius,
        results_df=df_results
    )

    return sim_state_frames


if __name__ == "__main__":
    
    # Configurations
    eval_config = load_config("examples/experiments/eval/config/eval_config")
    model_config = load_config("examples/experiments/eval/config/model_config")
    
    MODEL_TO_LOAD = "model_PPO__R_1000__01_19_11_15_25_854_002500"
 
    # Load policy
    policy = load_policy(
        path_to_cpt=model_config.models_path,
        model_name=MODEL_TO_LOAD,
        device=eval_config.device,
    )
    
    logging.info(f"Loaded policy {MODEL_TO_LOAD}")
    
    # Load results dataframe
    df_res = pd.read_csv(f"examples/experiments/eval/dataframes/0120/{MODEL_TO_LOAD}.csv")
    df_res = df_res[df_res['dataset'] == 'train']

    # Rollout policy and make videos
    sim_state_frames_train = make_videos(
        df_results=df_res, 
        policy=policy, 
        eval_config=eval_config,
        sort_by=None,
        show_top_k=50,
        device='cuda', 
        zoom_radius=100, 
        deterministic=False,
        render_every_n_steps=3,
    )

    sim_state_arrays_train = {
        k: np.array(v) for k, v in sim_state_frames_train.items()
    }
    
    videos_dir = Path(f"videos/{MODEL_TO_LOAD}")
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Save videos locally
    for env_id, frames in sim_state_arrays_train.items():
    
        video_path = videos_dir / f"train_scene_{env_id}.mp4"
        
        mediapy.write_video(
            str(video_path),
            frames,
            fps=5, 
        )
        
        logging.info(f"Saved video to {video_path}")