#!/usr/bin/env python3
"""
GPU Drive Environment Demo

This script demonstrates how to use the GPU Drive environment,
which provides a gymnasium-compatible interface to a driving simulator.
"""

import os
from pathlib import Path
import torch
import mediapy

# Import GPU Drive modules
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

# Environment settings
MAX_NUM_OBJECTS = 64  # Maximum number of objects in the scene we control
NUM_WORLDS = 2  # Number of parallel environments
UNIQUE_SCENES = 2  # Number of unique scenes
device = 'cpu'  # For simplicity we use CPU, but the simulator is optimized for GPU

def main():
    """Main function to run the environment demo."""
    
    # Initialize environment configuration
    # Here we define the discrete action space:
    # - steer_actions: 3 discrete steering actions from -1.0 to 1.0
    # - accel_actions: 3 discrete acceleration actions from -3.0 to 3.0
    env_config = EnvConfig(
        steer_actions=torch.round(torch.linspace(-1.0, 1.0, 3), decimals=3),
        accel_actions=torch.round(torch.linspace(-3, 3, 3), decimals=3)
    )

    # Create a data loader to load scenes
    # The data loader loads scene data from disk and provides it to the environment
    data_loader = SceneDataLoader(
        root="data/processed/examples",  # Path to the dataset
        batch_size=NUM_WORLDS,  # Batch size equals number of worlds so each world gets a different scene
        dataset_size=UNIQUE_SCENES,  # Total number of different scenes to use
        sample_with_replacement=False,
        seed=42,
        shuffle=True,   
    )

    # Create the environment
    # GPUDriveTorchEnv is a Gym-compatible environment that uses PyTorch tensors
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=MAX_NUM_OBJECTS,  # Maximum number of agents to control per scenario
        device=device,
    )

    # Print the data batch (scene files being used)
    print("Data batch (scene files):", env.data_batch)

    # Reset the environment to get initial observations
    obs = env.reset()

    # Initialize dictionary to store frames for visualization
    frames = {f"env_{i}": [] for i in range(NUM_WORLDS)}

    # Run a single rollout (one episode)
    for t in range(env_config.episode_len):
        # Sample random actions
        # For each object in each world, sample a random action from the action space
        rand_action = torch.Tensor(
            [[env.action_space.sample() for _ in range(MAX_NUM_OBJECTS * NUM_WORLDS)]]
        ).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)

        # Step the environment dynamics
        env.step_dynamics(rand_action)

        # Get observations, rewards, and done flags
        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

        # Render the environment every 5 steps
        if t % 5 == 0:
            imgs = env.vis.plot_simulator_state(
                env_indices=list(range(NUM_WORLDS)),
                time_steps=[t]*NUM_WORLDS,
                zoom_radius=70,
            )
        
            # Store the rendered images in the frames dictionary
            for i in range(NUM_WORLDS):
                frames[f"env_{i}"].append(img_from_fig(imgs[i]))
            
        # Break if all environments are done
        if done.all():
            break

    # Display videos of agents taking random actions
    # This will create a video for each environment
    mediapy.show_videos(frames, fps=5, width=500, height=500, columns=2, codec='gif')

if __name__ == "__main__":
    main()
    print("Demo completed!")