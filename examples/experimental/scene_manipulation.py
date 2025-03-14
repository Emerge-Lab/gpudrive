import os
import logging
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from eval_utils import load_policy, rollout, load_config, make_env

from pygpudrive.env.dataset import SceneDataLoader
from pygpudrive.datatypes.observation import LocalEgoState

import pdb
    
def test_policy_robustness(
    env, 
    policy,
    data_loader,
    config,
    remove_random_agents=False,
    remove_controlled_agents=True,
    plot_trajectories=False
):
    
    res_dict = {
        "scene": [],
        "goal_achieved": [],
        "collided": [],
        "off_road": [],
        "not_goal_nor_crashed": [],
        "controlled_agents_in_scene": [],
    }

    for batch in tqdm(
        data_loader,
        desc=f"Processing; remove_random_agents = {remove_random_agents}",
        total=len(data_loader),
        colour="red",
    ):
        # Set new data batch with simulator
        env.swap_data_batch(batch)
        
        # Sanity check: plot scenes before removing agents
        sim_state_figs_before = env.vis.plot_simulator_state(
            env_indices=[0, 1],
            time_steps=[0, 0],
        )
        sim_state_figs_before[0].savefig(f"sim_state_before_0.png")
        sim_state_figs_before[1].savefig(f"sim_state_before_1.png")
        
        if remove_random_agents:
            env.remove_agents_by_id(
                config.perc_to_rmv_per_scene,
                remove_controlled_agents = remove_controlled_agents
            )
        
        # Check after
        sim_state_figs_after = env.vis.plot_simulator_state(
            env_indices=[0, 1],
            time_steps=[0, 0],
        )
        sim_state_figs_after[0].savefig(f"sim_state_after_0.png")
        sim_state_figs_after[1].savefig(f"sim_state_after_1.png")
        
        # Rollout policy
        (
            goal_achieved,
            collided,
            off_road,
            controlled_agents_in_scene,
            not_goal_nor_crashed,
            _,
            agent_positions,
        ) = rollout(
            env=env,
            policy=policy,
            device=config.device,
            deterministic=config.deterministic,
            render_sim_state=config.render_sim_state,
            return_agent_positions=plot_trajectories
        )
        
        # Save last timestep rollout
        _ = env.reset()
        last_sim_state = env.vis.plot_simulator_state(
            env_indices=[0, 1],
            time_steps=[-1, -1],
            agent_positions=agent_positions
        )
        last_sim_state[0].savefig(f"last_sim_state_0.png")
        last_sim_state[1].savefig(f"last_sim_state_1.png")

        # Store results for the current batch
        scenario_names = [Path(path).stem for path in batch]
        res_dict["scene"].extend(scenario_names)
        res_dict["goal_achieved"].extend(goal_achieved.cpu().numpy())
        res_dict["collided"].extend(collided.cpu().numpy())
        res_dict["off_road"].extend(off_road.cpu().numpy())
        res_dict["not_goal_nor_crashed"].extend(
            not_goal_nor_crashed.cpu().numpy()
        )
        res_dict["controlled_agents_in_scene"].extend(
            controlled_agents_in_scene.cpu().numpy()
        )

    df_res = pd.DataFrame(res_dict)
    if remove_random_agents:
        df_res["deleted_agents"] = True
    else:
        df_res["deleted_agents"] = False
     
    return df_res

if __name__ == "__main__":
    
    config = load_config("examples/experimental/config/scene_manipulation_config")
    
    train_loader = SceneDataLoader(
        root=config.train_path,
        batch_size=config.num_worlds,
        dataset_size=config.dataset_size,
        sample_with_replacement=False,
    )
    
    # Make env
    env = make_env(config, train_loader)
        
    # Load policy
    policy = load_policy(
        path_to_cpt=config.cpt_path,
        model_name=config.cpt_name,
        device=config.device,
        env=env,
    ) 
    
    # Run tests
    df_perf_original = test_policy_robustness(
        env=env, 
        policy=policy,
        data_loader=train_loader,
        config=config,
        remove_random_agents=False,
        plot_trajectories=True
    )
    
    df_perf_perturbed_controlled = test_policy_robustness(
        env=env, 
        policy=policy,
        data_loader=train_loader,
        config=config,
        remove_random_agents=True,
        plot_trajectories=True        
    )

    df_perf_perturbed_static = test_policy_robustness(
        env=env,
        policy=policy,
        data_loader=train_loader,
        config=config,
        remove_random_agents=True,
        remove_controlled_agents=False,
        plot_trajectories=True
    )

    # Concatenate all three dataframes with a new column to identify the scenario
    df_perf_original['Class'] = 'Original'
    df_perf_perturbed_controlled['Class'] = 'Removed controlled'
    df_perf_perturbed_static['Class'] = 'Removed other'

    df = pd.concat([df_perf_original, df_perf_perturbed_controlled, df_perf_perturbed_static])

    # # Calculate mean values for each metric grouped by deleted_agents
    # metrics = ['goal_achieved', 'collided', 'off_road', 'not_goal_nor_crashed']

    # # Convert boolean columns to float for averaging
    # for col in metrics:
    #     df[col] = df[col].astype(float)

    # # Now calculate means
    # means_by_group = df.groupby('scenario')[metrics].mean()

    # # Set up the plot
    # fig, ax = plt.subplots(figsize=(12, 6))
    # x = np.arange(len(metrics))
    # width = 0.25

    # # Plot bars for each group
    # ax.bar(x - width, means_by_group.loc['Original'], width, 
    #     label='Original', color='skyblue')
    # ax.bar(x, means_by_group.loc['Removed Controlled'], width,
    #     label=f'Removed {int(config.perc_to_rmv_per_scene*100)}% Controlled', color='lightcoral')
    # ax.bar(x + width, means_by_group.loc['Removed Uncontrolled'], width,
    #     label=f'Removed {int(config.perc_to_rmv_per_scene*100)}% Uncontrolled', color='lightgreen')

    # # Customize the plot
    # ax.set_ylabel('Proportion')
    # ax.set_title('Policy Performance Comparison')
    # ax.set_xticks(x)
    # ax.set_xticklabels(metrics, rotation=45)
    # ax.legend()

    # # Adjust layout and save
    # plt.tight_layout()
    # plt.savefig(f"{config.save_results_path}/metrics_comparison_{int(config.perc_to_rmv_per_scene*100)}pct.png")
    # plt.close()
    
    # Save
    if not os.path.exists(config.save_results_path):
        os.makedirs(config.save_results_path)

    df.to_csv(f"{config.save_results_path}/combined_results_{int(config.perc_to_rmv_per_scene*100)}pct.csv", index=False)

    logging.info(f"Saved results at {config.save_results_path}")