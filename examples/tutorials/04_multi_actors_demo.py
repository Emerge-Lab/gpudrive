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
from pygpudrive.agents.expert_actor import HumanExpertActor
from pygpudrive.agents.policy_actor import PolicyActor
from pygpudrive.agents.core import merge_actions

if __name__ == "__main__":

    # Constants
    EPISODE_LENGTH = 90
    MAX_CONTROLLED_AGENTS = 128
    NUM_WORLDS = 3
    K_UNIQUE_SCENES = 10
    VIDEO_PATH = (
        "videos/multi_actors_demo_control_rand+policy_multiple_worlds.gif"
    )
    DEVICE = "cuda"
    DATA_PATH = "data"
    RENDER_WORLD_IDX = 2  # Scene we want to render

    # Configs
    env_config = EnvConfig()
    scene_config = SceneConfig(
        path=DATA_PATH,
        num_scenes=NUM_WORLDS,
        discipline=SelectionDiscipline.FIRST_N,
        k_unique_scenes=K_UNIQUE_SCENES,
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
    obj_idx = torch.arange(MAX_CONTROLLED_AGENTS)

    # rand_actor = RandomActor(
    #     env=env,
    #     is_controlled_func=(obj_idx == 0) | (obj_idx == 1),
    #     valid_agent_mask=env.cont_agent_mask,
    # )

    expert_actor = HumanExpertActor(
        is_controlled_func=(obj_idx < 3),
        valid_agent_mask=env.cont_agent_mask,
    )

    policy_actor = PolicyActor(
        is_controlled_func=obj_idx >= 3,
        valid_agent_mask=env.cont_agent_mask,
        saved_model_path="models/policy_23066479.zip",
        device=DEVICE,
    )

    obs = env.reset()
    frames = []

    # STEP THROUGH ENVIRONMENT
    for time_step in range(EPISODE_LENGTH):
        print(f"Step {time_step}/{EPISODE_LENGTH}")

        # SELECT ACTIONS
        # rand_actions = rand_actor.select_action()
        expert_actions = expert_actor.select_action(obs)
        rl_agent_actions = policy_actor.select_action(obs)

        # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
        actions = merge_actions(
            actor_actions_dict={
                # "pi_rand": rand_actions,
                "pi_rl": rl_agent_actions,
                "pi_expert": expert_actions,
            },
            actor_ids_dict={
                # "pi_rand": rand_actor.actor_ids,
                "pi_rl": policy_actor.actor_ids,
                "pi_expert": expert_actor.actor_ids,
            },
            reference_action_tensor=env.cont_agent_mask,
            device=DEVICE,
        )

        # STEP
        env.step_dynamics(actions)

        # GET NEXT OBS
        obs = env.get_obs()

        # RENDER
        frame = env.render(
            world_render_idx=RENDER_WORLD_IDX,
            color_objects_by_actor={
                # "rand": rand_actor.actor_ids[RENDER_WORLD_IDX].tolist(),
                "policy": policy_actor.actor_ids[RENDER_WORLD_IDX].tolist(),
                "expert": expert_actor.actor_ids[RENDER_WORLD_IDX].tolist(),
            },
        )
        frames.append(frame)

    print(f"Done. Saving video at {VIDEO_PATH}")
    imageio.mimwrite(VIDEO_PATH, np.array(frames), fps=15, loop=0)
