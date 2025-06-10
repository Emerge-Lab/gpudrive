#!/usr/bin/env python3
"""
GPUDrive Reward Conditioning Analysis

This script analyzes the effect of reward conditioning on agent behavior in GPUDrive,
including trajectory visualization and collision rate comparisons.
"""

import torch
import dataclasses
import mediapy
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from PIL import Image
import os
import sys

# GPUDrive imports
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

# Hugging Face imports
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet

_project_root = os.path.abspath("/home/mad10149/adaptive_driving_agent")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from gpudrive.examples.experimental.eval_utils import load_policy
from gpudrive.examples.experimental.eval_utils import rollout


def setup_environment(config_path, model_path, data_path, num_envs=4, device="cuda", max_agents=64):
    """
    Set up the GPUDrive environment and load the trained model.
    
    Args:
        config_path: Path to the configuration file
        model_path: Path to the trained model
        data_path: Path to the dataset
        num_envs: Number of environments to run in parallel
        device: Device to run on ('cpu' or 'cuda')
        max_agents: Maximum number of agents
        
    Returns:
        env: GPUDrive environment
        agent: Loaded policy
        config: Environment configuration
    """
    # Load configuration
    config = load_config(config_path)
    # print("Loaded configuration:")
    # print(config)
    
    # Set reward conditioning parameters
    config.environment.reward_type = "reward_conditioned"
    config.environment.condition_mode = "fixed"
    config.environment.agent_type = torch.Tensor([config.environment.collision_weight, config.environment.goal_achieved_weight, config.environment.off_road_weight])
    
    # Load the trained agent
    agent = load_policy(
        model_name="rew_conditioned_0321",
        path_to_cpt=model_path,
        env=config.environment,
        device=device
    )
    
    # Create data loader
    train_loader = SceneDataLoader(
        root=data_path,
        batch_size=num_envs,
        dataset_size=100,
        sample_with_replacement=False,
    )
    
    # Set environment parameters
    env_config = dataclasses.replace(
        EnvConfig(),
        norm_obs=config.environment.norm_obs,
        dynamics_model=config.environment.dynamics_model,
        collision_behavior=config.environment.collision_behavior,
        dist_to_goal_threshold=config.environment.dist_to_goal_threshold,
        polyline_reduction_threshold=config.environment.polyline_reduction_threshold,
        remove_non_vehicles=config.environment.remove_non_vehicles,
        lidar_obs=config.environment.lidar_obs,
        disable_classic_obs=config.environment.lidar_obs,
        obs_radius=config.environment.obs_radius,
        steer_actions=torch.round(
            torch.linspace(-torch.pi, torch.pi, config.environment.action_space_steer_disc), 
            decimals=3  
        ),
        accel_actions=torch.round(
            torch.linspace(-4.0, 4.0, config.environment.action_space_accel_disc), 
            decimals=3
        ),
        reward_type=config.environment.reward_type,
        condition_mode=config.environment.condition_mode,
        collision_weight=config.environment.collision_weight,
        goal_achieved_weight=config.environment.goal_achieved_weight,
        off_road_weight=config.environment.off_road_weight,
        agent_type=config.environment.agent_type,
    )
    
    # Create environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=1,
        device=device,
    )
    
    return env, agent, config


def run_multiple_rollouts(env, agent, num_rollouts=2, device='cpu',
                         sample_collision_weights=True, sample_goal_weights=False,
                         sample_offroad_weights=False, agent_type=None):
    """
    Run multiple rollouts with different collision weights and store trajectories.
    
    Args:
        env: The environment (can be batched with multiple environments)
        agent: The policy
        num_rollouts: Number of rollouts to perform
        device: Device to run on
        sample_collision_weights: Whether to sample collision weights
        sample_goal_weights: Whether to sample goal weights
        sample_offroad_weights: Whether to sample off-road weights
        agent_type: Fixed agent type weights (if provided)
        
    Returns:
        all_trajectories: Dictionary containing trajectories and weights
    """
    all_agent_positions = []
    collision_weights = []
    goal_weights = []
    offroad_weights = []
    all_goal_achieved = []
    all_collided = []
    all_off_road = []
    all_episode_lengths = []
    
    print(f"Running {num_rollouts} rollouts, sampling weights: "
          f"collision={sample_collision_weights}, goal={sample_goal_weights}, "
          f"offroad={sample_offroad_weights}\n")
    
    for i in tqdm(range(num_rollouts), desc="Processing rollouts", unit="rollout"):
        # Sample weights
        if agent_type is not None:
            agent_weights = agent_type
            collision_weight = agent_weights[0].item()
            goal_weight = agent_weights[1].item()
            off_road_weight = agent_weights[2].item()
        else:
            collision_weight = random.uniform(-3.0, 1.0) if sample_collision_weights else -3.0
            goal_weight = random.uniform(1.0, 3.0) if sample_goal_weights else 1.0
            off_road_weight = random.uniform(-3.0, 1.0) if sample_offroad_weights else -3.0
            agent_weights = torch.Tensor([collision_weight, goal_weight, off_road_weight])
        
        # Run rollout with these weights
        (goal_achieved_count, frac_goal_achieved, collided_count, frac_collided,
         off_road_count, frac_off_road, not_goal_nor_crash_count,
         frac_not_goal_nor_crash_per_scene, controlled_agents_per_scene,
         sim_state_frames, agent_positions, episode_lengths) = rollout(
            env=env,
            policy=agent,
            device=device,
            deterministic=False,
            return_agent_positions=True,
            set_agent_type=True,
            agent_weights=agent_weights,
        )
        
        # Store results
        collision_weights.append(collision_weight)
        goal_weights.append(goal_weight)
        offroad_weights.append(off_road_weight)
        all_agent_positions.append(agent_positions.clone().detach())
        all_goal_achieved.append(goal_achieved_count)
        all_collided.append(collided_count)
        all_off_road.append(off_road_count)
        all_episode_lengths.append(episode_lengths)
    
    # Stack agent positions along a new dimension
    stacked_positions = torch.stack(all_agent_positions, dim=1)
    
    # Return organized data
    all_trajectories = {
        'collision_weights': torch.tensor(collision_weights),
        'goal_weights': torch.tensor(goal_weights),
        'offroad_weights': torch.tensor(offroad_weights),
        'agent_positions': stacked_positions,
        'goal_achieved': all_goal_achieved,
        'collided': all_collided,
        'off_road': all_off_road,  
        'episode_lengths': all_episode_lengths
    }
    
    return all_trajectories


def visualize_trajectories(env, trajectories, output_filename="effect_of_rew_cond.png"):
    """
    Visualize agent trajectories with different reward conditioning.
    
    Args:
        env: GPUDrive environment
        trajectories: Dictionary containing trajectory data
        output_filename: Output filename for the plot
        
    Returns:
        Image: PIL Image of the visualization
    """
    env.vis.figsize = (8, 8)
    
    # Reset environment and plot
    _ = env.reset(agent_type=torch.Tensor([-0.2, 1.0, -0.2]))
    img = env.vis.plot_simulator_state(
        env_indices=[1], 
        agent_positions=trajectories['agent_positions'],
        zoom_radius=70,
        multiple_rollouts=True,
        line_alpha=0.5,          
        line_width=1.0,     
        weights=trajectories['collision_weights'],     
        colorbar=True, 
    )[0]
    
    # Save figure
    img.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure saved as {output_filename}")
    
    return Image.fromarray(img_from_fig(img))


def analyze_collision_rates():
    """
    Create collision rate comparison plots.
    """
    # Set up styling
    sns.set("notebook", font_scale=1.05, rc={"figure.figsize": (10, 5)})
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    
    # Prepare data
    agents = ['Self-play', 'Population play']
    all_agents_rates = [0.5, 2.85]
    single_agent_rates = [6.0, 9.50]
    
    # Create DataFrames
    all_agents_df = pd.DataFrame({
        'Agent': agents,
        'Collision Rate (%)': all_agents_rates
    })
    
    single_agent_df = pd.DataFrame({
        'Agent': agents,
        'Collision Rate (%)': single_agent_rates
    })
    
    # Calculate relative increases
    relative_increases = [single_agent_rates[i] / all_agents_rates[i] for i in range(len(agents))]
    relative_df = pd.DataFrame({
        'Agent': agents,
        'Performance drop SP -> Human': relative_increases
    })
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Controlling all agents
    sns.barplot(x='Agent', y='Collision Rate (%)', hue='Agent', data=all_agents_df, 
               palette='muted', legend=False, ax=axs[0])
    axs[0].set_title('Controlling All Agents')
    axs[0].set_ylim(0, max(all_agents_rates) * 1.2)
    for i, v in enumerate(all_agents_rates):
        axs[0].text(i, v + 0.1, f"{v}%", ha='center')
    
    # Plot 2: Controlling single agent
    sns.barplot(x='Agent', y='Collision Rate (%)', hue='Agent', data=single_agent_df, 
               palette='pastel', legend=False, ax=axs[1])
    axs[1].set_title('Control Single Agent \n with Human Replays')
    axs[1].set_ylim(0, max(single_agent_rates) * 1.2)
    for i, v in enumerate(single_agent_rates):
        axs[1].text(i, v + 0.3, f"{v}%", ha='center')
    
    # Plot 3: Relative increase
    sns.barplot(x='Agent', y='Performance drop SP -> Human', hue='Agent', data=relative_df, 
               palette='dark', legend=False, ax=axs[2])
    axs[2].set_title('Rel. Increase in Collision Rates \n (X times)', y=1.05)
    for i, v in enumerate(relative_increases):
        axs[2].text(i, v + 0.3, f"{v:.1f}x", ha='center')
    
    # Add annotation
    fig.text(0.5, 0.01, 
             "Note: The Population play agent type is set to be risk averse.", 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    sns.despine()
    plt.savefig('collision_rates_comparison.png', dpi=300)
    plt.show()


def main():
    """
    Main function to run the complete analysis.
    """
    # Configuration paths (update these to match your setup)
    config_path = "/home/mad10149/adaptive_driving_agent/gpudrive/baselines/ppo/config/ppo_base_puffer"
    model_path = "/home/mad10149/adaptive_driving_agent/models/ada"
    data_path = "/home/mad10149/adaptive_driving_agent/gpudrive/data/processed/examples/"
    
    # Set up environment and load model
    print("Setting up environment and loading model...")
    env, agent, config = setup_environment(
        config_path=config_path,
        model_path=model_path, 
        data_path=data_path,
        num_envs=4,
        device="cuda",
        max_agents=64
    )
    
    # Define different agent types for comparison
    agent_configs = {
        'Nominal': torch.tensor([-0.75, 1.0, -0.75]),
        'Aggressive': torch.tensor([0.0, 2.0, 0.0]),
        'Risk-averse': torch.tensor([-2.0, 0.5, -2.0]),
    }
    
    # Run multiple rollouts with different collision weights
    print("Running trajectory analysis...")
    trajectories = run_multiple_rollouts(
        env=env,
        agent=agent,
        num_rollouts=50,
        device='cuda',
        sample_collision_weights=True,
        sample_goal_weights=False,
        sample_offroad_weights=False,
        # agent_type=agent_configs['Risk-averse'],  # Uncomment to use fixed agent type
    )
    
    # Visualize trajectories
    print("Creating trajectory visualization...")
    img = visualize_trajectories(env, trajectories)
    
    # Analyze collision rates
    print("Creating collision rate analysis...")
    analyze_collision_rates()
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Number of rollouts: {len(trajectories['collision_weights'])}")
    print(f"Collision weight range: [{trajectories['collision_weights'].min():.2f}, "
          f"{trajectories['collision_weights'].max():.2f}]")
    print(f"Average collision weight: {trajectories['collision_weights'].mean():.2f}")
    print(f"Agent positions tensor shape: {trajectories['agent_positions'].shape}")
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()