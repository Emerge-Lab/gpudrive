import wandb
import numpy as np

from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    RenderMode,
    SceneConfig,
    SelectionDiscipline,
)
from pygpudrive.env.env_torch import GPUDriveTorchEnv

EPISODE_LENGTH = 91
NUM_WORLDS = 20


def run_episode_and_log(env, world_index):

    env.reset()

    frames = []
    for _ in range(EPISODE_LENGTH):
        frame = env.render(world_index)

        env.step_dynamics(actions=None)

        frames.append(frame)

    frames = np.array(frames)

    wandb.log(
        {
            f"world {world_index}": wandb.Video(
                np.moveaxis(frames, -1, 1),
                fps=20,
                format="gif",
            )
        }
    )


if __name__ == "__main__":

    env_config = EnvConfig()
    render_config = RenderConfig(render_mode=RenderMode.PYGAME_ABSOLUTE)

    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=SceneConfig(
            "formatted_json_v2_no_tl_train",
            NUM_WORLDS,
            SelectionDiscipline.FIRST_N,
        ),
        max_cont_agents=0,  # Step all vehicles in expert-control mode
        device="cuda",
        render_config=render_config,
    )

    run = wandb.init(project="gpudrive", group="make_videos")

    for world_index in range(NUM_WORLDS):
        run_episode_and_log(env, world_index)
