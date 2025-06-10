#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to use a pre-trained simulation agent from Hugging Face hub with the GPUDrive environment.
This script provides functionality to load a pre-trained driving agent and run it in a simulation.
"""

import torch
import dataclasses
import mediapy
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config


def main():
    """Main function to set up and run the simulation with the pre-trained agent."""
    
    ###############################
    #          CONFIGS            #
    ###############################
    
    # Load configuration parameters from the predefined config file
    config = load_config("examples/experimental/config/reliable_agents_params")
    print("Configuration parameters:", config)
    
    # Extract key parameters for our environment setup
    max_agents = config.max_controlled_agents  # Maximum number of agents that can be controlled
    num_envs = 2  # Number of parallel environments to run
    device = "cpu"  # Using CPU for notebook compatibility (change to "cuda" for GPU acceleration)
    
    ###############################
    #      LOAD PRETRAINED        #
    ###############################
    
    # Load the pre-trained simulation agent model from Hugging Face hub
    sim_agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")
    
    # Check the action dimension (13 steering angles x 7 acceleration values)
    print(f"Action dimension: {sim_agent.action_dim}")
    
    # Check the observation dimension (size of flattened observation vector)
    print(f"Observation dimension: {sim_agent.obs_dim}")
    
    # Load additional model information from the model card
    card = ModelCard.load("daphne-cornelisse/policy_S10_000_02_27")
    print(f"Model tags: {card.data.tags}")
    
    # To inspect model architecture, uncomment:
    # print(sim_agent)
    
    # To inspect model weights, uncomment:
    # print(sim_agent.state_dict())
    
    ###############################
    #    ENVIRONMENT CREATION     #
    ###############################
    
    # Create data loader for the environment
    train_loader = SceneDataLoader(
        root='data/processed/examples',
        batch_size=num_envs,
        dataset_size=100,
        sample_with_replacement=False,
    )
    
    # Set environment parameters by replacing default values in EnvConfig
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,                                       # Include ego state in observations
        road_map_obs=config.road_map_obs,                                 # Include road map in observations
        partner_obs=config.partner_obs,                                   # Include partner observations
        reward_type=config.reward_type,                                   # Type of reward function
        norm_obs=config.norm_obs,                                         # Normalize observations
        dynamics_model=config.dynamics_model,                             # Type of dynamics model
        collision_behavior=config.collision_behavior,                     # How to handle collisions
        dist_to_goal_threshold=config.dist_to_goal_threshold,             # Distance threshold for goal achievement
        polyline_reduction_threshold=config.polyline_reduction_threshold, # Threshold for polyline reduction
        remove_non_vehicles=config.remove_non_vehicles,                   # Whether to remove non-vehicles
        lidar_obs=config.lidar_obs,                                       # Include lidar in observations
        disable_classic_obs=config.lidar_obs,                             # Disable classic observations if lidar is enabled
        obs_radius=config.obs_radius,                                     # Observation radius
        steer_actions=torch.round(                                        # Discretized steering actions
            torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), 
            decimals=3
        ),
        accel_actions=torch.round(                                        # Discretized acceleration actions
            torch.linspace(-4.0, 4.0, config.action_space_accel_disc), 
            decimals=3
        ),
    )
    
    # Create the environment with the configured parameters
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=config.max_controlled_agents,
        device=device,
    )
    
    # Display loaded environment data files
    print(f"Loaded environment data: {env.data_batch}")
    
    ###############################
    #          ROLLOUT            #
    ###############################
    
    # Reset the environment to get initial observations
    next_obs = env.reset()
    
    # Get the mask for controllable agents
    control_mask = env.cont_agent_mask
    
    print(f"Observation shape: {next_obs.shape}")
    
    # Dictionary to store rendered frames for each environment
    frames = {f"env_{i}": [] for i in range(num_envs)}
    
    # Run the simulation for the full episode length
    for time_step in range(env.episode_len):
        print(f"\rStep: {time_step}", end="", flush=True)
        
        # Predict actions using the pre-trained agent
        action, _, _, _ = sim_agent(
            next_obs[control_mask], deterministic=False
        )
        
        # Create a template for all agents' actions and fill in the controlled agents
        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)
        
        # Step the environment dynamics
        env.step_dynamics(action_template)
        
        # Render the current state
        sim_states = env.vis.plot_simulator_state(
            env_indices=list(range(num_envs)),
            time_steps=[time_step]*num_envs,
            zoom_radius=70,
        )
        
        # Store rendered frames
        for i in range(num_envs):
            frames[f"env_{i}"].append(img_from_fig(sim_states[i]))
        
        # Get updated observations, rewards, done flags, and info
        next_obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()
        
        # Break if all environments are done
        if done.all():
            break
    
    # Close the environment
    env.close()
    
    # To save or display the animations, uncomment:
    # for env_id, frame_list in frames.items():
    #     mediapy.write_video(f"{env_id}_simulation.mp4", frame_list, fps=10)
    #     mediapy.show_video(frame_list, fps=10)


if __name__ == "__main__":
    main()