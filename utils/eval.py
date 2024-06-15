"""Methods to evaluate policies."""
import torch
import pandas as pd
from tqdm import tqdm

from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.config import EnvConfig

# Constansts
VEH_TYPE_ID = 7
TIMESTEPS = 90


def select_action(obs, env, eval_mode, policy=None):
    """Select action based on evaluation mode."""

    if eval_mode == "policy" and policy is not None:
        action = policy(obs)
    elif eval_mode == "random":
        action = env.action_space.sample()
    elif eval_mode == "expert-teleport":
        action = None
    else:
        raise ValueError(
            f"Invalid evaluation mode: {eval_mode} or policy is None."
        )

    return action


def run_episode(env, eval_mode, metrics, norm_scene_level=True, policy=None):
    """Run an episode."""

    episode_stats = torch.zeros((env.num_sims, len(metrics) + 1))

    # Reset environment
    obs = env.reset()

    for _ in range(TIMESTEPS):

        # Take action
        action = select_action(obs, env, eval_mode, policy)

        # Step environment
        obs, reward, done, info = env.step(action)

    # Use type to idenfify vehicles
    valid_veh_mask = info[:, :, 4] == VEH_TYPE_ID

    # Return episode stats
    for world_idx in range(env.num_sims):
        episode_stats[world_idx, :4] = info[world_idx, :, :][
            valid_veh_mask[world_idx, :]
        ][:, :4].sum(axis=0)

        episode_stats[world_idx, 4] = valid_veh_mask[world_idx, :].sum().item()

        if norm_scene_level:  # Divide by number of valid agents
            episode_stats[world_idx, :4] /= episode_stats[world_idx, 4]

    return episode_stats


def evaluate_policy(
    eval_mode,
    data_dir,
    max_controlled_agents=128,
    num_worlds=1,
    policy=None,
    num_episodes=100,
    device="cuda",
    norm_scene_level=False,
    metrics=[
        "off_road",
        "veh_collision",
        "non_veh_collision",
        "goal_reached",
    ],  # TODO: Add more metrics
):

    # Set policy mode
    if eval_mode == "policy" and policy is not None:
        # Evaluate policy
        pass

    elif eval_mode == "expert-teleport":
        max_controlled_agents = (
            0  # Ensure all agents are stepped in expert mode
        )

    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        num_worlds=num_worlds,
        max_cont_agents=max_controlled_agents,
        data_dir=data_dir,
        device=device,
    )

    # Storage
    episode_stats = torch.zeros((num_episodes, num_worlds, len(metrics) + 1))

    pbar = tqdm(range(num_episodes), colour="green")
    for episode_i in pbar:
        episode_stats[episode_i, :] = run_episode(
            env=env,
            eval_mode=eval_mode,
            metrics=metrics,
            norm_scene_level=norm_scene_level,
        )

        pbar.set_description(
            f"Evaluating in {eval_mode} mode for {num_episodes} episodes"
        )

    # Make dataframe
    df_performance = pd.DataFrame(
        episode_stats.flatten(start_dim=0, end_dim=1),
        columns=metrics + ["num_valid_agents"],
    )

    return df_performance


if __name__ == "__main__":

    env_config = EnvConfig(
        sample_method="first_n",
    )

    df_eval = evaluate_policy(
        eval_mode="expert-teleport",
        data_dir="formatted_json_v2_no_tl_train",
        num_episodes=1,
        num_worlds=1,
        norm_scene_level=True,
    )
