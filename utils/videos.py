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
from pygpudrive.env.scene_selector import select_scenes

EPISODE_LENGTH = 20
NUM_WORLDS = 20


import os
import numpy as np
import wandb
import shutil


def run_episode_and_log(env, world_index, file_name):

    env.reset()

    frames = []
    for _ in range(EPISODE_LENGTH):
        frame = env.render(world_index)

        env.step_dynamics(actions=None)

        frames.append(frame)

    frames = np.array(frames)

    # Check if the file name matches the specified one and save the JSON
    if world_index == 18:
        source_file_path = env.dataset[world_index]
        destination_file_path = os.path.join(
            ".", "tfrecord-00000-of-00150_238786b3cccc4a0e.json"
        )  # Save in the local folder

        # Save the raw JSON file by copying it
        shutil.copy(source_file_path, destination_file_path)
        print(f"Saved raw JSON data to {destination_file_path}")

    wandb.log(
        {
            f"world {world_index}": wandb.Video(
                np.moveaxis(frames, -1, 1),
                fps=20,
                format="gif",
                caption=f"File: {file_name}",  # Add the filename as a caption
            )
        }
    )


if __name__ == "__main__":

    env_config = EnvConfig()
    render_config = RenderConfig(render_mode=RenderMode.PYGAME_ABSOLUTE)
    scene_config = SceneConfig(
        "data/processed/validation",
        NUM_WORLDS,
        SelectionDiscipline.FIRST_N,
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=0,  # Step all vehicles in expert-control mode
        device="cuda",
        render_config=render_config,
    )

    run = wandb.init(project="pufferlib-integration", group="make_videos")

    for world_index in range(NUM_WORLDS):
        if world_index < len(env.dataset):
            file_name = env.dataset[world_index]
        else:
            file_name = "unknown_file"  # Fallback in case there are fewer files than worlds

        run_episode_and_log(env, world_index, file_name)
