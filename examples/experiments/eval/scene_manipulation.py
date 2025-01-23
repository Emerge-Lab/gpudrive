import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import torch
from evaluate import load_policy, rollout, load_config, make_env

from pygpudrive.env.dataset import SceneDataLoader
from pygpudrive.datatypes.observation import LocalEgoState

import pdb
    
def test_policy_robustness(
    env, 
    policy,
    data_loader,
    config,
    remove_random_agents=False,
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
            env.remove_agents_by_id(config.perc_to_rmv_per_scene)
        
        # Check after
        sim_state_figs_after = env.vis.plot_simulator_state(
            env_indices=[0, 1],
            time_steps=[0, 0]
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
        ) = rollout(
            env=env,
            policy=policy,
            device=config.device,
            deterministic=config.deterministic,
            render_sim_state=config.render_sim_state,
        )

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
    
    config = load_config("examples/experiments/eval/config/scene_manipulation_config")
    
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
    # df_perf_original = test_policy_robustness(
    #     env=env, 
    #     policy=policy,
    #     data_loader=train_loader,
    #     config=config,
    #     remove_random_agents=False,
    # )
    
    df_perf_perturbed = test_policy_robustness(
        env=env, 
        policy=policy,
        data_loader=train_loader,
        config=config,
        remove_random_agents=True,
    )

    df = pd.concat([df_perf_original, df_perf_perturbed])
    
    
    # Save
    if not os.path.exists(eval_config.res_path):
        os.makedirs(eval_config.res_path)

    df_res.to_csv(f"{config.save_results_path}/0123.csv", index=False)

    logging.info(f"Saved at {config.save_results_path}/0123.csv \n")

    


