import torch
import imageio
import numpy as np
from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
    SelectionDiscipline,
)
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.agents.random_actor import RandomActor
from pygpudrive.agents.policy_actor import PolicyActor
from pygpudrive.agents.core import merge_actions

if __name__ == "__main__":

    # Constants
    EPISODE_LENGTH = 90
    MAX_CONTROLLED_AGENTS = 6 # Number of agents to control per scene
    NUM_WORLDS = 100
    DEVICE = "cuda"
    DATA_PATH = "data"
    TRAINED_POLICY_PATH = "models/learned_sb3_policy.zip"
    VIDEO_PATH = (f"videos/release/")
    FPS = 23

    # Configs
    env_config = EnvConfig()
    scene_config = SceneConfig(
        path=DATA_PATH,
        num_scenes=NUM_WORLDS,
        discipline=SelectionDiscipline.FIRST_N,
    )
    render_config = RenderConfig(
        draw_obj_idx=True,
    )

    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        render_config=render_config,
        max_cont_agents=MAX_CONTROLLED_AGENTS,  # Maximum number of agents to control per scene
        device=DEVICE,
    )

    # Create sim agent 
    obj_idx = torch.arange(env_config.k_max_agent_count)

    # rand_actor = RandomActor(
    #     env=env,
    #     is_controlled_func=obj_idx < 3, #(obj_idx == 0) | (obj_idx == 1),
    #     valid_agent_mask=env.cont_agent_mask,
    # )
    
    policy_actor = PolicyActor(
        is_controlled_func=obj_idx >= 0,
        valid_agent_mask=env.cont_agent_mask,
        saved_model_path=TRAINED_POLICY_PATH,
        device=DEVICE,
    )
    
    obs = env.reset()
    
    frames_dict = {f'scene_{idx}': [] for idx in range(NUM_WORLDS)}

    # STEP THROUGH ENVIRONMENT
    for time_step in range(EPISODE_LENGTH):
        print(f"Step {time_step}/{EPISODE_LENGTH}")

        # SELECT ACTIONS
        #rand_actions = rand_actor.select_action()
        policy_actions = policy_actor.select_action(obs)

        # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
        actions = merge_actions(
            actor_actions_dict={
                #"pi_rand": rand_actions,
                "pi_rl": policy_actions,
            },
            actor_ids_dict={
                #"pi_rand": rand_actor.actor_ids,
                "pi_rl": policy_actor.actor_ids,
            },
            reference_action_tensor=env.cont_agent_mask,
            device=DEVICE,
        )

        # STEP
        env.step_dynamics(actions)

        # GET NEXT OBS
        obs = env.get_obs()

        # RENDER
        for world_idx in range(NUM_WORLDS):
            frame = env.render(
                world_render_idx=world_idx,
                color_objects_by_actor={
                    #"rand": rand_actor.actor_ids[world_idx].tolist(),
                    "policy": policy_actor.actor_ids[world_idx].tolist(),
                },
            )
            frames_dict[f'scene_{world_idx}'].append(frame)
    
    # # # # # # # #
    # Done. Save videos
    for scene_name, frames_list in frames_dict.items():
        frames_arr = np.array(frames_list)
        save_path = f"{VIDEO_PATH}{scene_name}.gif"
        print(f"Saving video at {save_path}")
        imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)
        
