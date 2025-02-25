import torch
import imageio
import numpy as np
from gpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
    SelectionDiscipline,
    RenderMode,
)
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.agents.random_actor import RandomActor
from gpudrive.agents.policy_actor import PolicyActor
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.agents.core import merge_actions
from examples.experimental.eval.eval_utils import load_policy
from gpudrive.visualize.utils import img_from_fig


def create_policy_masks(env, num_policies=2):
    policy_mask = torch.zeros_like(env.cont_agent_mask, dtype=torch.int)
    agent_indices = env.cont_agent_mask.nonzero(as_tuple=True)
    
    for i, (world_idx, agent_idx) in enumerate(zip(*agent_indices)):
        policy_mask[world_idx, agent_idx] = (i % num_policies) + 1
    
    policy_mask = {f'pi_{int(policy.item())}': (policy_mask == policy)
            for policy in policy_mask.unique() if policy.item() != 0}
    

    policy_world_mask = {
        world: {f'pi_{p+1}': policy_mask[f'pi_{p+1}'][world] for p in range(NUM_POLICIES)}
        for world in range(env.cont_agent_mask.shape[0])
    }
    return policy_world_mask

if __name__ == "__main__":

    # Constants
    EPISODE_LENGTH = 30
    MAX_CONTROLLED_AGENTS = 64 # Number of agents to control per scene
    NUM_WORLDS = 2
    DEVICE = "cpu"
    DATA_PATH = "data/processed/examples" 
    TRAINED_POLICY_PATH = "trained_policies"
    TRAINED_POLICY_NAME1 = "model_PPO____S_10__02_14_12_52_18_771_000300"
    TRAINED_POLICY_NAME2 = "model_PPO____S_10__02_14_18_43_12_828_001700"
    VIDEO_PATH = f"videos/"
    FPS = 23
    UNIQUE_SCENES = 2 

    # Configs
    env_config = EnvConfig()
    # Make dataloader
    data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=NUM_WORLDS, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=UNIQUE_SCENES, # Total number of different scenes we want to use
    sample_with_replacement=False, 
    seed=42, 
    shuffle=True,   
    )

    # Make environment
    env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=data_loader,
    max_cont_agents=MAX_CONTROLLED_AGENTS, # Maximum number of agents to control per scenario
    device=DEVICE,
    )


    # Create sim agent
    obj_idx = torch.arange(env.max_cont_agents)

    # rand_actor = RandomActor(
    #     env=env,
    #     is_controlled_func=(obj_idx == 0), #(obj_idx == 0) | (obj_idx == 1),
    #     valid_agent_mask=env.cont_agent_mask,
    #     device=DEVICE
    # )


    policy1 = load_policy(TRAINED_POLICY_PATH,TRAINED_POLICY_NAME1,DEVICE,env=env)
    policy2 = load_policy(TRAINED_POLICY_PATH,TRAINED_POLICY_NAME2,DEVICE,env=env)

    policy_actor1 = PolicyActor(
        is_controlled_func=(obj_idx == 1),
        valid_agent_mask=env.cont_agent_mask,
        policy=policy1,
        device=DEVICE,
    )

    policy_actor2 = PolicyActor(
        is_controlled_func=(obj_idx == 1),
        valid_agent_mask=env.cont_agent_mask,
        policy=policy2,
        device=DEVICE,
    )
    NUM_POLICIES = 2


    policy_masks= create_policy_masks(env,NUM_POLICIES)


    obs = env.reset(env.cont_agent_mask)

    frames_dict = {f"scene_{idx}": [] for idx in range(NUM_WORLDS)}



    # STEP THROUGH ENVIRONMENT
    for time_step in range(EPISODE_LENGTH):
        print(f"Step {time_step}/{EPISODE_LENGTH}")

        # SELECT ACTIONS
        actions1 = policy_actor1.select_action(obs)
        actions2 = policy_actor2.select_action(obs)



        actor_actions_dict = {
                "pi_1": actions1,
                "pi_2": actions2,

                }
        
        actor_ids_dict={
                "pi_1": policy_actor1.actor_ids,
                "pi_2": policy_actor2.actor_ids,
            }
        

                

        # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
        actions = merge_actions(
            actor_actions_dict=actor_actions_dict,
            reference_action_tensor=env.cont_agent_mask,
            policy_masks=policy_masks,
            device=DEVICE,
        )

        ## map actions ussing maks
        # STEP
        env.step_dynamics(actions)

        # GET NEXT OBS
        obs = env.get_obs(env.cont_agent_mask)


        ## RENDER 
        if time_step % 5 == 0:
            imgs = env.vis.plot_simulator_state(
                env_indices=list(range(NUM_WORLDS)),
                time_steps=[time_step]*NUM_WORLDS,
                zoom_radius=70,
                policy_mask=policy_masks
            )

            for i in range(NUM_WORLDS):
                frames_dict[f"scene_{i}"].append(img_from_fig(imgs[i])) 




    for scene_name, frames_list in frames_dict.items():
        frames_arr = np.array(frames_list)
        save_path = f"{VIDEO_PATH}{scene_name}.gif"
        print(f"Saving video at {save_path}")
        try:
            imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)
        except:
            import os
            os.mkdir(VIDEO_PATH)
            imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)


