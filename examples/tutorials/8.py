#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiple Policies Script for GPUDrive

This script demonstrates how to load and run multiple policy models in the GPUDrive 
environment. It creates a simulation with different agents controlled by different 
policies and renders the simulation to visualize agent behaviors.
"""

import torch
import dataclasses
import mediapy
import numpy as np
import os
import sys
import imageio
from huggingface_hub import PyTorchModelHubMixin, ModelCard
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.utils.multi_policy_rollout import multi_policy_rollout


def create_policy_masks(env, num_sim_agents=2, num_worlds=10):
    """
    Create masks for assigning different policies to different agents.
    
    Args:
        env: The GPUDrive environment
        num_sim_agents: Number of different policies to use
        num_worlds: Number of simulation worlds
        
    Returns:
        Dictionary of policy masks, where each mask is a boolean tensor that selects
        which agents should be controlled by that policy
    """
    policy_mask = torch.zeros_like(env.cont_agent_mask, dtype=torch.int)
    agent_indices = env.cont_agent_mask.nonzero(as_tuple=True)

    # Assign policy IDs to each agent
    for i, (world_idx, agent_idx) in enumerate(zip(*agent_indices)):
        policy_mask[world_idx, agent_idx] = (i % num_sim_agents) + 1

    # Create a mask for each policy
    policy_masks = {
        f'pi_{int(policy.item())}': torch.zeros_like(env.cont_agent_mask, dtype=torch.bool, device=device) 
        for policy in policy_mask.unique() if policy.item() != 0
    }

    # Set the mask values for each policy
    for p in range(1, num_sim_agents + 1):
        policy_masks[f'pi_{p}'] = (policy_mask == p).reshape(num_worlds, -1)

    return policy_masks


def main():
    """Main function to setup and run the simulation"""
    global device  # Make device available to create_policy_masks

    # Load configuration that the models were trained with
    config = load_config("../../examples/experimental/config/reliable_agents_params")
    max_agents = config.max_controlled_agents
    
    # Simulation parameters
    NUM_ENVS = 2        # Number of parallel environments to simulate
    device = "cpu"      # Using CPU for notebook compatibility
    NUM_SIM_AGENTS = 2  # Number of different policies
    FPS = 5             # Frames per second for rendering
    
    # Load pre-trained policy models
    sim_agent1 = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")
    sim_agent2 = NeuralNet.from_pretrained("daphne-cornelisse/policy_S1000_02_27")
    
    # Get model information (optional)
    card = ModelCard.load("daphne-cornelisse/policy_S10_000_02_27")
    
    # Initialize data loader for simulation scenarios
    train_loader = SceneDataLoader(
        root='../../data/processed/examples',
        batch_size=NUM_ENVS,
        dataset_size=100,
        sample_with_replacement=False,
    )
    
    # Configure environment parameters
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        norm_obs=config.norm_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        dist_to_goal_threshold=config.dist_to_goal_threshold,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=config.lidar_obs,
        disable_classic_obs=config.lidar_obs,
        obs_radius=config.obs_radius,
        steer_actions=torch.round(
            torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), 
            decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-4.0, 4.0, config.action_space_accel_disc), 
            decimals=3
        ),
    )
    
    # Create environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=max_agents,
        device=device,
    )
    
    # Reset environment and get initial observation
    next_obs = env.reset()
    control_mask = env.cont_agent_mask
    
    # Create masks to assign different policies to different agents
    policy_mask = create_policy_masks(env, NUM_SIM_AGENTS, NUM_ENVS)
    
    # Create a dictionary mapping policy names to (policy model, mask) tuples
    policies_set = {
        'pi_1': (sim_agent1, policy_mask['pi_1']),
        'pi_2': (sim_agent2, policy_mask['pi_2'])
    } 
    
    # Run simulation with multiple policies
    metrics, frames = multi_policy_rollout(
        env,
        policies_set, 
        device,
        deterministic=False,
        render_sim_state=True,
        render_every_n_steps=5
    )
    
    # Clean up environment resources
    env.close()
    
    # Display simulation videos
    mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')


if __name__ == "__main__":
    main()