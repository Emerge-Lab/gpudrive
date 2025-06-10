#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPUDrive Simulator Interface

This script demonstrates how to work with the GPUDrive simulator and access
its basic attributes in Python. The simulator, written in C++, is built on 
top of the Madrona Engine.

Key concepts:
- GPUDrive simulations are discretized traffic scenarios
- A scenario is a constructed snapshot of traffic at a particular timepoint
- The state of the vehicle of focus is referred to as the "ego state"
- Each vehicle has their own partial view of the traffic scene
- The visible state is constructed by parameterizing the view distance
"""

import os
import torch
from pathlib import Path
import madrona_gpudrive

# -----------------------------------------------------------------------------
# Setup working directory
# -----------------------------------------------------------------------------
def setup_working_directory():
    """Set working directory to the base 'gpudrive' directory."""
    working_dir = Path.cwd()
    while working_dir.name != 'gpudrive':
        working_dir = working_dir.parent
        if working_dir == Path.home():
            raise FileNotFoundError("Base directory 'gpudrive' not found")
    os.chdir(working_dir)
    print(f"Working directory set to: {working_dir}")

# -----------------------------------------------------------------------------
# Initialize simulator
# -----------------------------------------------------------------------------
def initialize_simulator(scene_path=None):
    """
    Initialize the GPUDrive simulator with default or custom parameters.
    
    Args:
        scene_path (str, optional): Path to the scene file. Defaults to an example scene.
    
    Returns:
        SimManager: Initialized simulator instance
    """
    # Set device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set default scene path if not provided
    if not scene_path:
        scene_path = "data/processed/examples/tfrecord-00000-of-01000_325.json"
    
    # Initialize parameters with custom settings
    params = madrona_gpudrive.Parameters()
    params.polylineReductionThreshold = 0.1
    params.observationRadius = 100.0
    params.collisionBehaviour = madrona_gpudrive.CollisionBehaviour.Ignore
    params.maxNumControlledAgents = 10
    
    # Configure reward parameters
    reward_params = madrona_gpudrive.RewardParams()
    reward_params.rewardType = madrona_gpudrive.RewardType.DistanceBased
    reward_params.distanceToGoalThreshold = 1.0
    reward_params.distanceToExpertThreshold = 1.0
    params.rewardParams = reward_params
    
    # Create simulator instance
    sim = madrona_gpudrive.SimManager(
        exec_mode=madrona_gpudrive.madrona.ExecMode.CUDA
        if device == "cuda"
        else madrona_gpudrive.madrona.ExecMode.CPU,
        gpu_id=0,
        scenes=[scene_path],
        params=params,
    )
    
    return sim

# -----------------------------------------------------------------------------
# Simulator operations
# -----------------------------------------------------------------------------
def reset_worlds(sim, world_indices=None):
    """
    Reset specified worlds in the simulator.
    
    Args:
        sim: The simulator instance
        world_indices (list, optional): List of world indices to reset. Defaults to [0].
    """
    if world_indices is None:
        world_indices = [0]
    sim.reset(world_indices)
    print(f"Reset worlds: {world_indices}")

def step_simulator(sim):
    """
    Advance the simulation by one step.
    
    Args:
        sim: The simulator instance
    """
    sim.step()
    print("Advanced simulation by one step")

# -----------------------------------------------------------------------------
# Data access functions
# -----------------------------------------------------------------------------
def get_observation_tensors(sim):
    """
    Get observation tensors from the simulator.
    
    Args:
        sim: The simulator instance
        
    Returns:
        dict: Dictionary containing different observation tensors
    """
    tensors = {
        "self_observation": sim.self_observation_tensor().to_torch(),
        "map_observation": sim.map_observation_tensor().to_torch(),
        "partner_observations": sim.partner_observations_tensor().to_torch(),
        "agent_roadmap": sim.agent_roadmap_tensor().to_torch()
    }
    
    for name, tensor in tensors.items():
        print(f"{name} shape: {tensor.shape}, device: {tensor.device}")
    
    return tensors

def get_controlled_agents_info(sim):
    """
    Get information about which agents can be controlled.
    
    Args:
        sim: The simulator instance
        
    Returns:
        torch.Tensor: Tensor of controlled agents (1 = controllable, 0 = not controllable)
    """
    controlled_state_tensor = sim.controlled_state_tensor().to_torch()
    print(f"Controlled state tensor shape: {controlled_state_tensor.shape}")
    print(f"Number of controllable agents: {controlled_state_tensor.sum().item()}")
    
    return controlled_state_tensor

def set_random_actions(sim):
    """
    Set random actions for all agents in the simulator.
    
    Args:
        sim: The simulator instance
        
    Returns:
        torch.Tensor: Tensor of random actions that were applied
    """
    actions_tensor = sim.action_tensor().to_torch()
    random_actions = torch.rand(actions_tensor.shape)
    actions_tensor.copy_(random_actions)
    
    print(f"Set random actions with shape: {random_actions.shape}")
    return random_actions

def get_simulation_state(sim):
    """
    Get the current state of the simulation (observations, rewards, done flags).
    
    Args:
        sim: The simulator instance
        
    Returns:
        tuple: (observations, rewards, done flags)
    """
    obs = sim.self_observation_tensor().to_torch()
    rewards = sim.reward_tensor().to_torch()
    dones = sim.done_tensor().to_torch()
    
    print(f"Observations shape: {obs.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Done flags shape: {dones.shape}")
    
    return obs, rewards, dones

def print_parameters(params=None):
    """
    Print all parameters of the simulator.
    
    Args:
        params: Parameter object. If None, creates a new default Parameters object.
    """
    if params is None:
        params = madrona_gpudrive.Parameters()
    
    print("\nSimulator Parameters:")
    print("=" * 50)
    for attr in dir(params):
        if not attr.startswith("__"):
            value = getattr(params, attr)
            print(f"{attr:20}: {value}")
            if attr == "rewardParams":
                print("\nReward Parameters:")
                print("-" * 50)
                reward_params = getattr(params, attr)
                for attr2 in dir(reward_params):
                    if not attr2.startswith("__"):
                        value2 = getattr(reward_params, attr2)
                        print(f"    {attr2:18}: {value2}")

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    """Main function demonstrating the use of the GPUDrive simulator."""
    # Setup working directory
    try:
        setup_working_directory()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Continuing without changing directory...")
    
    # Initialize simulator
    sim = initialize_simulator()
    
    # Reset the first world
    reset_worlds(sim)
    
    # Get information about controlled agents
    controlled_agents = get_controlled_agents_info(sim)
    
    # Get observation tensors
    observation_tensors = get_observation_tensors(sim)
    
    # Set random actions for all agents
    random_actions = set_random_actions(sim)
    
    # Step the simulator
    step_simulator(sim)
    
    # Get simulation state after stepping
    obs, rewards, dones = get_simulation_state(sim)
    
    # Print simulator parameters
    print_parameters()
    
    print("\nSimulation demonstration complete.")

if __name__ == "__main__":
    main()