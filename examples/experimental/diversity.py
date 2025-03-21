import torch
import dataclasses
import mediapy
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.utils.checkpoint import load_policy
from gpudrive.utils.rollout import rollout

from PIL import Image


def collect_rollout_for_agent_type(
    env, agent, agent_type_name, agent_weights, device, num_rollouts=5
):
    """
    Collect rollout data for a specific agent type.

    Args:
        env: The simulation environment
        agent: The policy to be rolled out
        agent_type_name: Name identifier for this agent type
        agent_weights: Tensor of agent weights
        device: The device to run on
        num_rollouts: Number of rollouts to perform for this agent type

    Returns:
        list: List of dictionaries containing rollout data for this agent type
    """
    all_data = []

    # Run multiple rollouts for this agent type
    for i in tqdm(
        range(num_rollouts), desc=f"Agent: {agent_type_name}", unit="rollout"
    ):
        # Run the rollout with the specified agent weights
        rollout_results = rollout(
            env=env,
            policy=agent,
            device=device,
            deterministic=False,
            render_sim_state=False,
            return_agent_positions=False,
            return_behavior_metrics=True,
            set_agent_type=True,
            agent_weights=agent_weights,
        )

        # Extract behavioral metrics from rollout (last element of returned tuple)
        behavior_metrics = rollout_results[-1]

        # Store data for this rollout
        all_data.append(
            {
                "agent_type": agent_type_name,
                "rollout_idx": i,
                "weights": agent_weights.cpu().tolist(),
                "entropy": behavior_metrics["entropy"],
                "logprob": behavior_metrics["logprob"],
                "actions": behavior_metrics["actions"],
            }
        )

    return all_data


def collect_data_for_agent_types(
    env, agent, agent_configs, device, num_rollouts_per_agent=5
):
    """
    Collect rollout data for all agent types in agent_configs.

    Args:
        env: The simulation environment
        agent: The policy to be rolled out
        agent_configs: Dictionary mapping agent type names to their weight tensors
        device: The device to run on
        num_rollouts_per_agent: Number of rollouts to perform for each agent type

    Returns:
        list: Combined list of dictionaries containing rollout data for all agent types
    """
    all_agent_data = []

    # Display overall progress information
    print(
        f"Collecting data for {len(agent_configs)} agent types, {num_rollouts_per_agent} rollouts each"
    )

    # Process each agent type
    for agent_type_name, agent_weights in agent_configs.items():
        print(
            f"\nAgent type: {agent_type_name}, weights: {agent_weights.cpu().tolist()}"
        )

        # Collect data for this agent type
        agent_data = collect_rollout_for_agent_type(
            env=env,
            agent=agent,
            agent_type_name=agent_type_name,
            agent_weights=agent_weights,
            device=device,
            num_rollouts=num_rollouts_per_agent,
        )

        # Add to combined data
        all_agent_data.extend(agent_data)

    print(f"Completed rollouts for all {len(agent_configs)} agent types")
    return all_agent_data


def create_dataframe_from_rollouts(rollout_data):
    """
    Convert rollout data to a pandas DataFrame with the specified columns.

    Args:
        rollout_data: List of dictionaries containing rollout results

    Returns:
        pd.DataFrame: DataFrame with agent behavior data
    """
    rows = []

    # Process each rollout
    for data in rollout_data:
        agent_type = data["agent_type"]
        rollout_idx = data["rollout_idx"]
        weights = data["weights"]

        # Get tensor data
        entropy = data["entropy"]  # Shape: [num_envs, max_agents, episode_len]
        logprob = data["logprob"]  # Shape: [num_envs, max_agents, episode_len]
        actions = data[
            "actions"
        ]  # Shape: [num_envs, max_agents, episode_len, action_dim]

        # Get dimensions
        num_envs = actions.shape[0]
        max_agents = actions.shape[1]
        episode_len = actions.shape[2]

        # Create rows for all non-zero entries
        for env_idx in range(num_envs):
            for agent_idx in range(max_agents):
                for time_step in range(episode_len):
                    # Skip if no action was taken (all zeros)
                    action_vec = actions[env_idx, agent_idx, time_step]
                    if (
                        torch.all(action_vec == 0)
                        and entropy[env_idx, agent_idx, time_step] == 0
                    ):
                        continue

                    # Add row with specified columns
                    row = {
                        "agent_type": agent_type,
                        "scenario": env_idx,
                        "agent_idx": agent_idx,  # Keep this in case you need it later
                        "timestep": time_step,
                        "entropy": entropy[
                            env_idx, agent_idx, time_step
                        ].item(),
                        "acceleration": action_vec[0].item(),
                        "steering": action_vec[1].item(),
                        "logprob": logprob[
                            env_idx, agent_idx, time_step
                        ].item(),
                        "rollout_idx": rollout_idx,
                        "collision_weight": weights[0],
                        "goal_weight": weights[1],
                        "off_road_weight": weights[2],
                    }
                    rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Print summary
    print(f"Created DataFrame with {len(df)} rows")
    print(f"Number of unique agent types: {df['agent_type'].nunique()}")
    print(f"Number of unique scenarios: {df['scenario'].nunique()}")
    print(f"Maximum timestep: {df['timestep'].max()}")

    # Print more detailed statistics
    print("\nRows per agent type:")
    print(df.groupby("agent_type").size())

    print("\nSample of DataFrame:")
    print(df.head())

    return df


def compare_agent_types(df):
    """
    Compare different agent types based on their behavior using the updated DataFrame structure.

    Args:
        df: DataFrame containing agent behavior data with the columns:
            agent_type, scenario, agent_idx, timestep, entropy, acceleration, steering, etc.

    Returns:
        dict: Dictionary of analysis results
    """
    results = {}

    # 1. Compare acceleration and steering distributions by agent type
    accel_dist_by_agent = (
        df.groupby("agent_type")["acceleration"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    results["acceleration_distribution"] = accel_dist_by_agent

    steer_dist_by_agent = (
        df.groupby("agent_type")["steering"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    results["steering_distribution"] = steer_dist_by_agent

    # 2. Compare entropy (uncertainty) by agent type
    entropy_by_agent = df.groupby("agent_type")["entropy"].agg(
        ["mean", "std", "min", "max"]
    )
    results["entropy_stats"] = entropy_by_agent

    # 3. Create a measure of behavior diversity for each agent type
    # Higher entropy and more diverse action choices indicate more diverse behavior

    # Calculate action diversity based on unique combinations of acceleration and steering
    df["action_combo"] = (
        df["acceleration"].astype(str) + "_" + df["steering"].astype(str)
    )
    df_grouped = (
        df.groupby(["agent_type", "scenario", "timestep"])["action_combo"]
        .nunique()
        .reset_index()
    )
    action_diversity = df_grouped.groupby("agent_type")["action_combo"].mean()

    # Calculate entropy diversity
    entropy_diversity = entropy_by_agent["mean"]

    # Combine measures into overall behavior diversity
    behavior_diversity = pd.DataFrame(
        {
            "mean_entropy": entropy_by_agent["mean"],
            "action_diversity": action_diversity,
        }
    )

    # Normalize and combine (handle case where min = max)
    entropy_range = (
        behavior_diversity["mean_entropy"].max()
        - behavior_diversity["mean_entropy"].min()
    )
    action_range = (
        behavior_diversity["action_diversity"].max()
        - behavior_diversity["action_diversity"].min()
    )

    if entropy_range > 1e-10:
        behavior_diversity["norm_entropy"] = (
            behavior_diversity["mean_entropy"]
            - behavior_diversity["mean_entropy"].min()
        ) / entropy_range
    else:
        behavior_diversity["norm_entropy"] = 0.5  # Constant value if all equal

    if action_range > 1e-10:
        behavior_diversity["norm_action_div"] = (
            behavior_diversity["action_diversity"]
            - behavior_diversity["action_diversity"].min()
        ) / action_range
    else:
        behavior_diversity[
            "norm_action_div"
        ] = 0.5  # Constant value if all equal

    behavior_diversity["overall_diversity"] = (
        behavior_diversity["norm_entropy"]
        + behavior_diversity["norm_action_div"]
    ) / 2

    results["behavior_diversity"] = behavior_diversity

    # 4. Calculate additional metrics
    # Average logprob by agent type (measure of confidence)
    logprob_by_agent = df.groupby("agent_type")["logprob"].mean()
    results["avg_logprob"] = logprob_by_agent

    # Distribution of steering and acceleration over time
    time_series = df.groupby(["agent_type", "timestep"])[
        ["steering", "acceleration"]
    ].mean()
    results["time_series"] = time_series

    # Print summary of comparison
    print("\n=== Agent Type Comparison ===")
    print("\nBehavioral Diversity (higher = more diverse):")
    print(
        behavior_diversity[
            ["mean_entropy", "action_diversity", "overall_diversity"]
        ].sort_values("overall_diversity", ascending=False)
    )

    print("\nAverage Log Probability (higher = more confident):")
    print(logprob_by_agent)

    # Acceleration and steering tendencies
    print("\nAverage Acceleration by Agent Type:")
    print(
        df.groupby("agent_type")["acceleration"]
        .mean()
        .sort_values(ascending=False)
    )

    print(
        "\nAverage Steering by Agent Type (absolute value - measures turning intensity):"
    )
    print(
        df.groupby("agent_type")
        .apply(lambda x: abs(x["steering"]).mean())
        .sort_values(ascending=False)
    )

    # Analyze variation across scenarios
    scenario_variation = (
        df.groupby(["agent_type", "scenario"])[["acceleration", "steering"]]
        .mean()
        .groupby("agent_type")
        .std()
    )
    print("\nVariation Across Scenarios (higher = less consistent behavior):")
    print(scenario_variation)

    return results


if __name__ == "__main__":

    config = load_config(
        "/home/emerge/gpudrive/baselines/ppo/config/ppo_base_puffer"
    )

    num_envs = 20
    device = "cpu"
    max_agents = 2

    config.environment.reward_type = "reward_conditioned"
    config.environment.condition_mode = "fixed"
    config.environment.agent_type = torch.Tensor([-0.2, 1.0, -0.2])

    agent = load_policy(
        model_name="model_PPO____R_10000__03_20_17_01_59_614_007000",
        path_to_cpt="/home/emerge/gpudrive/examples/experimental/models",
        env_config=config.environment,
        device=device,
    )

    # Create data loader
    train_loader = SceneDataLoader(
        root="/home/emerge/gpudrive/data/processed/training/",
        batch_size=num_envs,
        dataset_size=100,
        sample_with_replacement=False,
    )

    # Set params
    config = config.environment
    env_config = dataclasses.replace(
        EnvConfig(),
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
            torch.linspace(
                -torch.pi, torch.pi, config.action_space_steer_disc
            ),
            decimals=3,
        ),
        accel_actions=torch.round(
            torch.linspace(-4.0, 4.0, config.action_space_accel_disc),
            decimals=3,
        ),
        reward_type=config.reward_type,
        condition_mode=config.condition_mode,
        agent_type=config.agent_type,
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=max_agents,
        device=device,
    )

    # Define different agent types to compare
    agent_configs = {  # Collision, Goal, Off-road
        "Nominal": torch.tensor([-0.75, 1.0, -0.75], device=device),
        "Aggressive": torch.tensor([0.0, 2.0, 0.0], device=device),
        "Risk-averse": torch.tensor([-2.0, 0.5, -2.0], device=device),
    }

    print("Collecting rollout data for all agent types...")

    # Collect data for all agent types
    all_rollout_data = collect_data_for_agent_types(
        env=env,
        agent=agent,
        agent_configs=agent_configs,
        device=device,
        num_rollouts_per_agent=10,  # Number of times we rollout in the same scenes
    )

    # Convert to DataFrame for analysis
    df = create_dataframe_from_rollouts(all_rollout_data)

    comparison_results = compare_agent_types(df)

    df.to_csv("agent_type_comparison_data.csv", index=False)
