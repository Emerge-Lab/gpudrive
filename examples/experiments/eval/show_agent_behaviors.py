import pandas as pd
from box import Box
import numpy as np
from pathlib import Path
import mediapy

from pygpudrive.env.dataset import SceneDataLoader
from eval_utils import load_policy, load_config, make_env, rollout

import logging

logging.basicConfig(level=logging.INFO)

import pdb

def make_videos(
    policy,
    eval_config,
    df_results=None,
    sort_by=None,  # Options: 'goal_achieved',
    show_top_k=100,
    device="cuda",
    zoom_radius=100,
    deterministic=False,
    render_every_n_steps=10,
):
    """Make videos policy rollouts environment.
    Args:
        df_results (pd.DataFrame): Dataframe with the results of the policy rollout.
        policy (torch.nn.Module): Policy to select actions.
        eval_config (Box): Configuration for the evaluation.
        sort_by (str): Sample scenarios from dataframe sorted by this column. Options:
            - goal_achieved: Sort by goal_achieved in descending order (successes first).
            - collided: Sort by collided in descending order.
            - off_road: Sort by off_road in descending order.
            - not_goal_nor_crashed: Sort by not_goal_nor_crashed in descending order.
            - controlled_agents_in_scene: Sort by controlled_agents_in_scene in descending order.
    """
    if df_results is not None:
        base_data_path = (
            eval_config.train_dir
            if df_results["dataset"][0] == "train"
            else eval_config.test_dir
        )
    else:
        base_data_path = eval_config.train_dir

    # Make environment
    train_loader = SceneDataLoader(
        root=base_data_path,
        batch_size=show_top_k,
        dataset_size=show_top_k,
        sample_with_replacement=False,
        shuffle=False,
    )
    
    env = make_env(eval_config, train_loader)

    # Select data batch toi
    if sort_by is not None and sort_by in df_results.columns:
        df_top_k = df_results.sort_values(by=sort_by, ascending=False).head(
            show_top_k
        )
        data_batch = (
            base_data_path + "/" + df_top_k.scene.values + ".json"
        ).tolist()

    elif sort_by is None:
        data_batch = env.data_batch

    env.swap_data_batch(data_batch)

    # Rollout policy in the environments
    _, _, _, _, _, sim_state_frames, global_agent_states = rollout(
        env=env,
        policy=policy,
        device=device,
        deterministic=deterministic,
        render_sim_state=True,
        render_every_n_steps=render_every_n_steps,
        zoom_radius=zoom_radius,
        results_df=df_results,
    )

    return sim_state_frames


if __name__ == "__main__":

    # Specify which model to load and the dataset to evaluate
    MODEL_TO_LOAD = "model_PPO__R_10000__01_23_21_02_58_770_007000" #"model_PPO__R_10000__01_23_21_02_58_770_005500"
    DATASET = "test"
    SORT_BY = "collided" #"goal_achieved"
    SHOW_TOP_K = 25 # Render this many scenes

    # Configurations
    eval_config = load_config("examples/experiments/eval/config/eval_config")
    model_config = load_config("examples/experiments/eval/config/model_config")

    # Load policy
    policy = load_policy(
        path_to_cpt=model_config.models_path,
        model_name=MODEL_TO_LOAD,
        device=eval_config.device,
    )

    # Load results dataframe
    if SORT_BY is not None:
        df_res = pd.read_csv(f"{eval_config.res_path}/{MODEL_TO_LOAD}.csv")
        df_res = df_res[df_res["dataset"] == DATASET]
    else:
        df_res = None
        
    logging.info(
        f"Loaded policy {MODEL_TO_LOAD} and corresponding results df."
    )

    # Rollout policy and make videos
    sim_state_frames = make_videos(
        df_results=df_res,
        policy=policy,
        eval_config=eval_config,
        sort_by=SORT_BY,  # Options: 'goal_achieved', 'collided', 'off_road', 'not_goal_nor_crashed', 'controlled_agents_in_scene'
        show_top_k=SHOW_TOP_K,
        device="cuda",
        zoom_radius=100,
        deterministic=False,
        render_every_n_steps=3,
    )

    sim_state_arrays = {k: np.array(v) for k, v in sim_state_frames.items()}
    
    SORT_BY = SORT_BY if not None else "random"
    videos_dir = Path(f"videos/{MODEL_TO_LOAD}/{SORT_BY}_top_{SHOW_TOP_K}")
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Save videos locally
    for env_id, frames in sim_state_arrays.items():

        video_path = videos_dir / f"scene_{env_id}.mp4"

        mediapy.write_video(
            str(video_path),
            frames,
            fps=10,
        )

        logging.info(f"Saved video to {video_path}")
