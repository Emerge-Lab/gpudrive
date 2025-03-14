import pandas as pd
from box import Box
import numpy as np
from pathlib import Path
import mediapy

from gpudrive.env.dataset import SceneDataLoader
from eval_utils import load_policy, load_config, make_env, rollout

import logging

logging.basicConfig(level=logging.INFO)

import pdb
import random
import torch
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)
SEED = 42  
set_seed(SEED)

def make_videos(
    policy,
    eval_config,
    df_results=None,
    dataset="train",
    sort_by=None,  # Options: 'goal_achieved',
    show_top_k=100,
    device="cuda",
    zoom_radius=100,
    deterministic=False,
    render_every_n_steps=10,
    center_on_ego=False,
    render_3d=True,
):
    """Make videos policy rollouts environment.
    Args:
        df_results (pd.DataFrame): Dataframe with the results of the policy rollout.
        policy (torch.nn.Module): Policy to select actions.
        eval_config (Box): Configuration for the evaluation.
        sort_by (str): Sample scenarios from dataframe sorted by this column. Options:
            - goal_achieved_frac: Sort by goal_achieved in descending order (successes first).
            - collided_frac: Sort by collided in descending order.
            - off_road_frac: Sort by off_road in descending order.
            - other_frac: Sort by not_goal_nor_crashed in descending order.
            - controlled_agents_in_scene: Sort by controlled_agents_in_scene in descending order.
    """
    
    base_data_path = (
        eval_config.train_dir if dataset == "train"
        else eval_config.test_dir
    )
  
    if df_results is not None:
        df_results = df_results[df_results["dataset"] == dataset]

    # Make environment
    train_loader = SceneDataLoader(
        root=base_data_path,
        batch_size=show_top_k,
        dataset_size=show_top_k,
        sample_with_replacement=False,
        shuffle=False,
    )
    
    env = make_env(eval_config, train_loader, render_3d)

    # Select data batch to
    if sort_by is not None and sort_by in df_results.columns:
        df_top_k = df_results.sort_values(by=sort_by, ascending=False).head(
            show_top_k
        )
        data_batch = (
            base_data_path + "/" + df_top_k.scene.values
        ).tolist()

    elif sort_by is None:
        data_batch = env.data_batch

    env.swap_data_batch(data_batch)

    # Rollout policy in the environments
    _, _, _, _, _, _, _, _, _, sim_state_frames, global_agent_states, _ = rollout(
        env=env,
        policy=policy,
        device=device,
        deterministic=deterministic,
        render_sim_state=True,
        render_every_n_steps=render_every_n_steps,
        zoom_radius=zoom_radius,
        center_on_ego=center_on_ego,
    )

    return sim_state_frames, env.get_env_filenames()


if __name__ == "__main__":

    # Specify which model to load and the dataset to evaluate
    MODEL_TO_LOAD = "model_PPO____S_1000__02_26_08_54_58_359_009200"
    DATASET = "train"
    SORT_BY = None #"goal_achieved_frac" 
    SHOW_TOP_K = 50 # Render this many scenes

    # Configurations
    eval_config = load_config("examples/experimental/config/eval_config")
    model_config = load_config("examples/experimental/config/model_config")

    # Load policy
    policy = load_policy(
        path_to_cpt=model_config.models_path,
        model_name=MODEL_TO_LOAD,
        device=eval_config.device,
    )

    # Load results dataframe
    if SORT_BY is not None:
        df_res = pd.read_csv(f"{eval_config.res_path}/{MODEL_TO_LOAD}.csv")
        df_res = df_res[df_res['controlled_agents_in_scene'] > 3]
    else:
        df_res = None
    
    logging.info(
        f"Loaded policy {MODEL_TO_LOAD} and corresponding results df."
    )

    # Rollout policy and make videos
    sim_state_frames, filenames = make_videos(
        df_results=df_res,
        policy=policy,
        eval_config=eval_config,
        sort_by=SORT_BY,  # Options: 'goal_achieved', 'collided', 'off_road', 'not_goal_nor_crashed', 'controlled_agents_in_scene'
        show_top_k=SHOW_TOP_K,
        dataset=DATASET,
        device=eval_config.device,
        zoom_radius=40,
        deterministic=False,
        render_every_n_steps=1,
        #center_on_ego=True,
        render_3d=True,
    )

    sim_state_arrays = {k: np.array(v) for k, v in sim_state_frames.items()}
    
    SORT_BY = SORT_BY if not None else "random"
    videos_dir = Path(f"videos/{MODEL_TO_LOAD}/project_page") # Path(f"videos/{MODEL_TO_LOAD}/{SORT_BY}_top_{SHOW_TOP_K}")
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Save videos locally
    for env_id, frames in sim_state_arrays.items():
        
        filename = filenames[env_id]
        
        if df_res:
            scene_stats = df_res[df_res["scene"] == filename]
            goal_achieved = scene_stats.goal_achieved_frac.values.item()
            collided = scene_stats.collided_frac.values.item()
            off_road = scene_stats.off_road_frac.values.item()
            other = scene_stats.other_frac.values.item()

            video_path = videos_dir / f"{filename}_ga_{goal_achieved:.2f}__cr_{collided:.2f}__or_{off_road:.2f}__ot_{other:.2f}.gif"
        else:
            video_path = videos_dir / f"{filename}.gif"

        mediapy.write_video(
            str(video_path),
            frames,
            fps=17,
            codec='gif',
        )

        logging.info(f"Saved video to {video_path}")
