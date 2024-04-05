"""Vectorized environment wrapper for multi-agent environments."""
import logging
from copy import deepcopy
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

# Import base gumenvironment
from pygpudrive.env.base_environment import Env

logging.basicConfig(level=logging.INFO)


class SB3MultiAgentEnv(VecEnv):
    """Casts multi-agent environments as vectorized environments.

    Args:
    -----
        VecEnv (SB3 VecEnv): SB3 VecEnv base class.
    """

    def __init__(
        self, config, num_worlds, max_cont_agents, data_dir, device, auto_reset=True
    ):
        self.env = Env(
            config=config,
            num_worlds=num_worlds,
            max_cont_agents=max_cont_agents,
            data_dir=data_dir,
            device=device,
            auto_reset=auto_reset,
        )
        self.num_worlds = num_worlds
        self.max_cont_agents = max_cont_agents
        self.num_envs = self.num_worlds * self.max_cont_agents
        self.device = device
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self.env.observation_space.shape, np.float32
        )
        self.obs_dim = self.env.observation_space.shape[0]

    def _reset_seeds(self) -> None:
        """Reset all environments' seeds."""
        self._seeds = None

    def reset(self, seed=None):
        """Reset environment and return initial observations.

        Returns:
        --------
            torch.Tensor (max_cont_agents * num_worlds, obs_dim): Initial observations.
        """

        # Has shape (num_worlds, max_cont_agents, obs_dim)
        obs = self.env.reset()

        # Flatten over num_worlds and max_cont_agents
        obs = obs.reshape(self.num_envs, self.obs_dim)

        # Save observation to buffer
        self._save_obs(obs)

        # Get alive agent mask
        self.dead_agent_mask = torch.full(
            (self.num_worlds, self.max_cont_agents), fill_value=False
        ).to(self.device)

        return self._obs_from_buf()

    def step(self, actions) -> VecEnvStepReturn:
        """
        Returns:
        --------
            torch.Tensor (max_cont_agents * num_worlds, obs_dim): Observations.
            torch.Tensor (max_cont_agents * num_worlds): Rewards.
            torch.Tensor (max_cont_agents * num_worlds): Dones.
            dict: Additional information.
        """

        # Unsqueeze action tensor to a shape gpu drive expects
        actions = actions.reshape((self.num_worlds, self.max_cont_agents))

        # Step the environment
        obs, reward, done, info = self.env.step(actions)

        print(f"done: {done}\n")
        print(f"dead_agent_mask: {self.dead_agent_mask}\n")

        # Update dead agent mask (Set to True if agent is done before end of episode)
        self.dead_agent_mask = torch.logical_or(self.dead_agent_mask, done)

        # Storage: Fill buffer with nan values
        self.buf_rews = torch.full(
            (self.num_worlds, self.max_cont_agents), fill_value=float("nan")
        ).to(self.device)
        self.buf_dones = torch.full(
            (self.num_worlds, self.max_cont_agents), fill_value=float("nan")
        ).to(self.device)
        self.buf_infos = torch.full(
            (self.num_worlds, self.max_cont_agents), fill_value=float("nan")
        ).to(self.device)
        buf_obs = torch.full(
            (self.num_worlds, self.max_cont_agents, self.obs_dim),
            fill_value=float("nan"),
        ).to(self.device)

        # Override nan placeholders for alive agents
        self.buf_rews[~self.dead_agent_mask] = reward[~self.dead_agent_mask]
        self.buf_dones[~self.dead_agent_mask] = done[~self.dead_agent_mask].to(
            torch.float32
        )
        self.buf_infos[~self.dead_agent_mask] = info[~self.dead_agent_mask].to(
            torch.float32
        )
        buf_obs[~self.dead_agent_mask] = obs[~self.dead_agent_mask]

        # Flatten over num_worlds and max_cont_agents and store
        obs = obs.reshape(self.num_envs, self.obs_dim)
        self._save_obs(obs)

        # Reset episode if all agents in all worlds are done
        if torch.all(self.dead_agent_mask):
            print(f"Resetting environment\n")
            obs = self.reset()

        return (
            self._obs_from_buf(),
            torch.clone(self.buf_rews).reshape(self.num_envs),
            torch.clone(self.buf_dones).reshape(self.num_envs),
            deepcopy(self.buf_infos).reshape(self.num_envs),
        )

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def seed(self, seed=None):
        """Set the random seeds for all environments."""
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(
                np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            )

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def _save_obs(self, obs: VecEnvObs) -> None:
        """Save observations into buffer."""
        self.buf_obs = obs

    def _obs_from_buf(self) -> VecEnvObs:
        """Get observation from buffer."""
        return self.buf_obs.clone()

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(
        self, method_name, *method_args, indices=None, **method_kwargs
    ):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()


if __name__ == "__main__":

    # Make environment
    env = SB3MultiAgentEnv(
        num_worlds=1,
        max_cont_agents=2,
        data_dir="waymo_data",
        device="cuda",
        auto_reset=True,
    )

    obs = env.reset()
    for global_step in range(100):

        # Random actions
        actions = torch.randint(0, env.action_space.n, (env.num_envs,))

        # Step
        obs, rew, done, info = env.step(actions)
