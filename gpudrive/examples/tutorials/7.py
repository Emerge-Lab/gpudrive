#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent Behavior Diversity Through Reward Conditioning

This script demonstrates how to use reward conditioning to create diverse
agent behaviors in the GPUDrive environment. The technique allows for creating
a spectrum of behaviors from cautious to aggressive, all using a single policy network.

Inspired by:
    "Robust autonomy emerges from self-play" (Appendix B.3)
    https://arxiv.org/abs/2502.03349
"""

import torch
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.torch_env import GPUDriveTorchEnv


def main():
    """
    Main function to demonstrate different reward conditioning modes
    """
    # Step 1: Create environment configuration
    env_config = EnvConfig(
        # Enable reward conditioning by setting this parameter
        reward_type="reward_conditioned",
        
        # Set reward bounds for each component
        # These bounds define the range of possible rewards for different behaviors
        # Collision penalty (negative values)
        collision_weight_lb=-1.0,  # Lower bound (more negative = stronger penalty)
        collision_weight_ub=-0.1,  # Upper bound (less negative = weaker penalty)
        
        # Goal achievement reward (positive values)
        goal_achieved_weight_lb=0.5,  # Lower bound (smaller reward)
        goal_achieved_weight_ub=2.0,  # Upper bound (larger reward)
        
        # Off-road penalty (negative values)
        off_road_weight_lb=-1.0,  # Lower bound (more negative = stronger penalty)
        off_road_weight_ub=-0.1,  # Upper bound (less negative = weaker penalty)
        
        # Default conditioning mode during training
        condition_mode="random"
    )
    
    # Create render configuration with default settings
    render_config = RenderConfig()
    
    # Step 2: Create a data loader for scene data
    train_loader = SceneDataLoader(
        root="data/processed/examples",
        batch_size=2,
        dataset_size=100,
        sample_with_replacement=True,
        shuffle=False,
    )
    
    # Step 3: Create the GPUDrive environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=64,  # Number of agents to control
        device="cuda",  # Use GPU if available, change to "cpu" if no GPU
    )
    
    # Get controlled agent mask (identifies which agents can be controlled)
    control_mask = env.cont_agent_mask
    
    # Step 4: Reset the environment (using the default random conditioning mode)
    obs = env.reset()
    print(f"Environment reset with RANDOM conditioning mode (default)")
    
    # Run a brief simulation with random conditioning
    run_simulation(env, num_steps=10)
    
    # Step 5: Demonstrate preset conditioning mode with different agent profiles
    preset_types = ["cautious", "aggressive", "balanced", "risk_taker"]
    
    for agent_type in preset_types:
        print(f"\nResetting environment with PRESET conditioning mode: {agent_type}")
        obs = env.reset(condition_mode="preset", agent_type=agent_type)
        
        # Run a brief simulation with preset conditioning
        run_simulation(env, num_steps=10)
    
    # Step 6: Demonstrate fixed conditioning mode with custom weights
    print("\nResetting environment with FIXED conditioning mode (custom weights)")
    
    # Define custom weights [collision_weight, goal_weight, off_road_weight]
    custom_weights = torch.tensor([-0.75, 1.5, -0.3])
    obs = env.reset(condition_mode="fixed", agent_type=custom_weights)
    
    # Run a brief simulation with fixed conditioning
    run_simulation(env, num_steps=10)


def run_simulation(env, num_steps=10):
    """
    Run a brief simulation to demonstrate agent behavior with current conditioning
    
    Args:
        env: The GPUDrive environment
        num_steps: Number of steps to run the simulation
    """
    # Generate random actions for demonstration purposes
    # In a real application, actions would come from your policy network
    action_shape = env.action_space.shape
    
    for step in range(num_steps):
        # Generate random actions
        actions = torch.rand(action_shape, device=env.device) * 2 - 1
        
        # Take a step in the environment
        obs, rewards, done, info = env.step(actions)
        
        # Print rewards to observe conditioning effects
        if step == 0:
            # Print reward weights from the first controlled agent
            controlled_idx = torch.where(env.cont_agent_mask)[0][0].item()
            
            # Extract reward weights (these are included in the observations when using reward conditioning)
            # The last 3 elements of ego_state contain the reward weights
            ego_state = obs["ego_state"][controlled_idx]
            reward_weights = ego_state[-3:]
            
            print(f"  Agent reward weights: collision={reward_weights[0]:.2f}, "
                  f"goal={reward_weights[1]:.2f}, off_road={reward_weights[2]:.2f}")
    
    print(f"  Simulation complete: {num_steps} steps")


if __name__ == "__main__":
    main()