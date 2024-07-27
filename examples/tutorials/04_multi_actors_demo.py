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
    NUM_WORLDS = 1
    K_UNIQUE_SCENES = 1
    VIDEO_PATH = "videos/multi_actors_demo_control_3_different.gif"
    SCENE_NAME = "example_scene"
    DEVICE = "cuda"
    DATA_PATH = "data"

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

    # MAKE ENV
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        render_config=render_config,
        max_cont_agents=MAX_CONTROLLED_AGENTS,  # Maximum number of agents to control per scene
        device=DEVICE,
    )

    # CREATE SIM AGENTS
    obj_idx = torch.arange(MAX_CONTROLLED_AGENTS)

    rand_actor = RandomActor(
        env=env, is_controlled_func=(obj_idx == 0)  # | (obj_idx == 1),
    )

    expert_actor = HumanExpertActor(
        is_controlled_func=(obj_idx == 1),
    )

    policy_actor = PolicyActor(
        is_controlled_func=obj_idx > 1,
        saved_model_path="models/policy_23066479.zip",
    )

    obs = env.reset()
    frames = []

    # STEP THROUGH ENVIRONMENT
    for time_step in range(EPISODE_LENGTH):

        # SELECT ACTIONS
        rand_actions = rand_actor.select_action()
        expert_actions = expert_actor.select_action(obs)
        rl_agent_actions = policy_actor.select_action(obs)

        # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
        actions = merge_actions(
            actions={
                "pi_rand": rand_actions,
                "pi_rl": rl_agent_actions,
                "pi_expert": expert_actions,
            },
            actor_ids={
                "pi_rand": rand_actor.actor_ids,
                "pi_rl": policy_actor.actor_ids,
                "pi_expert": expert_actor.actor_ids,
            },
            reference_actor_shape=obj_idx,
        )

        # STEP
        env.step_dynamics(actions.reshape(1, MAX_CONTROLLED_AGENTS))

        # GET NEXT OBS
        obs = env.get_obs()

        # RENDER
        frame = env.render(
            world_render_idx=0,
            color_objects_by_actor={
                "rand": rand_actor.actor_ids.tolist(),
                "policy": policy_actor.actor_ids.tolist(),
                "expert": expert_actor.actor_ids.tolist(),
            },
        )
        frames.append(frame)

    print(f"Done. Saving video at {VIDEO_PATH}")
    imageio.mimwrite(VIDEO_PATH, np.array(frames), fps=30)
