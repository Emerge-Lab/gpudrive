import os
import logging
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import torch
from eval_utils import load_policy, rollout, load_config, make_env, evaluate_policy

from pygpudrive.env.dataset import SceneDataLoader
from pygpudrive.datatypes.observation import LocalEgoState
import pdb
    
    
if __name__ == "__main__":
    
    config = load_config("examples/experimental/config/hand_designed_experiments")
    
    # Load original scenes
    data_loader_orig = SceneDataLoader(
        root=config.data_path_original,
        batch_size=config.num_worlds,
        dataset_size=config.dataset_size,
        sample_with_replacement=False,
    )
    
    # Load altered scenes
    data_loader_altered = SceneDataLoader(
        root=config.data_path_altered,
        batch_size=config.num_worlds,
        dataset_size=config.dataset_size,
        sample_with_replacement=False,
    )
    
    # Make env
    env = make_env(config, data_loader_orig)
        
    # Load policy
    policy = load_policy(
        path_to_cpt=config.cpt_path,
        model_name=config.cpt_name,
        device=config.device,
        env=env,
    ) 
    
    # Run tests
    df_perf_original = evaluate_policy(
        env=env,
        policy=policy,
        data_loader=data_loader_orig,
        dataset_name="test",
        deterministic=False,
        render_sim_state=False,
    )
    
    df_perf_altered = evaluate_policy(
        env=env,
        policy=policy,
        data_loader=data_loader_altered,
        dataset_name="test",
        deterministic=False,
        render_sim_state=False,
    )
   
    # Concatenate all three dataframes with a new column to identify the scenario
    df_perf_original['Class'] = 'Original'
    df_perf_altered['Class'] = 'Altered'

    df = pd.concat([df_perf_original, df_perf_altered])

    metrics = ['goal_achieved_frac', 'collided_frac', 'off_road_frac', 'other_frac']

    tab_agg_perf = df.groupby('Class')[metrics].agg(['mean', 'std'])
    tab_agg_perf = tab_agg_perf * 100
    tab_agg_perf = tab_agg_perf.round(1)
        
    print('')  
    print(tab_agg_perf)
    print('')  
    
    # Save
    if not os.path.exists(config.save_results_path):
        os.makedirs(config.save_results_path)

    df.to_csv(f"{config.save_results_path}/combined_results_ood.csv", index=False)

    logging.info(f"Saved results at {config.save_results_path}")
    
    # # Make videos
    # videos_dir = Path(f"videos/{config.cpt_name}/hand_designed")
    # videos_dir.mkdir(parents=True, exist_ok=True)
    
    # for data_loader in [data_loader_orig]: #data_loader_altered
    
    #     for batch in tqdm(
    #         data_loader,
    #         desc=f"Making videos",
    #         total=len(data_loader),
    #         colour="MAGENTA",
    #     ):

    #         env.swap_data_batch(batch)
            
    #         (
    #             goal_achieved_count,
    #             goal_achieved_frac,
    #             collided_count,
    #             collided_frac,
    #             off_road_count,
    #             off_road_frac,
    #             other_count,
    #             other_frac,
    #             controlled_agents_in_scene,
    #             sim_state_frames,
    #             agent_positions,
    #             episode_lengths,
    #         ) = rollout(
    #             env=env,
    #             policy=policy,
    #             device=config.device,
    #             deterministic=config.device,
    #             render_sim_state=config.render_sim_state,
    #             render_every_n_steps=1,
    #             zoom_radius=config.zoom_radius,
    #         )
            
    #         filenames = env.get_env_filenames()
            
    #         sim_state_arrays = {k: np.array(v) for k, v in sim_state_frames.items()}

    #         # Save videos locally
    #         for env_id, frames in sim_state_arrays.items():
                
    #             filename = filenames[env_id]
                
    #             video_path = videos_dir / f"{filename}.mp4"

    #             mediapy.write_video(
    #                 str(video_path),
    #                 frames,
    #                 fps=15,
    #             )

    #             logging.info(f"Saved video to {video_path}")

        
        