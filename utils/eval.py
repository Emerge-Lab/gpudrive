"""Methods to evaluate policies."""
import torch
import pandas as pd
from tqdm import tqdm

from pygpudrive.env.base_environment import Env
from pygpudrive.env.config import EnvConfig


def run_episode(env, eval_mode, policy=None):
    """Run an episode."""

    # Reset environment
    obs = env.reset()

    for _ in range(env.steps_remaining):

        # Take action
        action = select_action(obs, env, eval_mode, policy)

        # Step environment
        obs, reward, done, info = env.step(action)

    # Return statistics
    episode_stats = info.sum(axis=1).squeeze().tolist()
    episode_stats.append(0)  # TODO: num_valid_agents

    return torch.Tensor(episode_stats)


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


def evaluate_policy(
    eval_mode,
    data_dir,
    max_controlled_agents=128,
    num_worlds=1,
    policy=None,
    num_episodes=100,
    device="cuda",
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
    env = Env(
        config=env_config,
        num_worlds=num_worlds,
        max_cont_agents=max_controlled_agents,
        data_dir=data_dir,
        device=device,
    )

    # Storage
    episode_stats = torch.zeros((num_episodes, len(metrics) + 1))

    pbar = tqdm(range(num_episodes), colour="green")
    for episode_i in pbar:
        episode_stats[episode_i, :] = run_episode(env, eval_mode)
        pbar.set_description(
            f"Evaluating in {eval_mode} mode for {num_episodes} episodes"
        )

    # Make dataframe
    df_performance = pd.DataFrame(
        episode_stats,
        columns=metrics + ["num_valid_agents"],
    )

    return df_performance


if __name__ == "__main__":

    env_config = EnvConfig(
        eval_expert_mode=True,
        sample_method="first_n",
    )

    df_eval = evaluate_policy(
        eval_mode="expert-teleport",
        data_dir="formatted_json_v2_no_tl_train",
    )
