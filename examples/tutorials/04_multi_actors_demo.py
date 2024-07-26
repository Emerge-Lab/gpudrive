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

if __name__ == "__main__":

    # Constants
    EPISODE_LENGTH = 90
    MAX_CONTROLLED_AGENTS = 1
    NUM_WORLDS = 1
    K_UNIQUE_SCENES = 1
    VIDEO_PATH = "videos/multi_actors_demo_control_one_agent.mp4"
    SCENE_NAME = "example_scene"
    DEVICE = "cuda"

    # Configs
    env_config = EnvConfig(
        steer_actions=torch.round(torch.linspace(-0.3, 0.3, 3), decimals=3),
        accel_actions=torch.Tensor([-20]),
    )
    scene_config = SceneConfig(
        path="example_data",
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

    obs = env.reset()
    frames = []

    # STEP THROUGH ENVIRONMENT
    for time_step in range(EPISODE_LENGTH):
        print(f"Step: {time_step}")

        # SELECT ACTIONS
        actions = torch.Tensor(
            [
                [
                    env.action_space.sample()
                    for _ in range(MAX_CONTROLLED_AGENTS * NUM_WORLDS)
                ]
            ]
        ).reshape(NUM_WORLDS, MAX_CONTROLLED_AGENTS)

        # STEP
        env.step_dynamics(actions)

        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

        # RENDER
        frame = env.render(world_render_idx=0)
        frames.append(frame)

    imageio.mimwrite(VIDEO_PATH, np.array(frames), fps=30)


actions = None
for time_step in range(EPISODE_LENGTH):

    # STEP
    env.step_dynamics(actions)

    obs = env.get_obs()
    reward = env.get_rewards()
    done = env.get_dones()

    # RENDER
    frame = env.render(world_render_idx=0)
    frames.append(frame)
