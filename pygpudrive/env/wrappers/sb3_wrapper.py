"""Vectorized environment wrapper for multi-agent environments."""
import logging
from typing import Optional, Sequence
import torch
from typing import Any, Dict, List, Optional, Sequence
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

# Import base gumenvironment
from pygpudrive.env.base_environment import Env

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig

logging.basicConfig(level=logging.INFO)


class SB3MultiAgentEnv(VecEnv):
    """Casts multi-agent environments as vectorized environments.

    Args:
    -----
        VecEnv (SB3 VecEnv): SB3 VecEnv base class.
    """

    def __init__(
        self,
        config,
        num_worlds,
        max_cont_agents,
        data_dir,
        device,
        render_mode="rgb_array",
    ):
        self._env = Env(
            config=config,
            num_worlds=num_worlds,
            max_cont_agents=max_cont_agents,
            data_dir=data_dir,
            device=device,
        )
        self.num_worlds = num_worlds
        self.max_agent_count = self._env.max_agent_count
        self.num_envs = self.num_worlds * self.max_agent_count
        self.device = device
        self.controlled_agent_mask = self._env.cont_agent_mask
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self._env.observation_space.shape, np.float32
        )
        self.obs_dim = self._env.observation_space.shape[0]
        self.info_dim = self._env.info_dim
        self.render_mode = render_mode

        self.num_episodes = 0
        self.infos = {
            "off_road": 0,
            "veh_collisions": 0,
            "non_veh_collision": 0,
            "goal_achieved": 0,
        }

    def _reset_seeds(self) -> None:
        """Reset all environments' seeds."""
        self._seeds = None

    def reset(self, seed=None):
        """Reset environment and return initial observations.

        Returns:
        --------
            torch.Tensor (max_agent_count * num_worlds, obs_dim):
                Initial observation.
        """

        # Has shape (num_worlds, max_agent_count , obs_dim)
        obs = self._env.reset()

        # Make dead agent mask (True for dead or invalid agents)
        self.dead_agent_mask = torch.isnan(obs[:, :, 0]).to(self.device)

        # Flatten over num_worlds and max_agent_count
        obs = obs.reshape(self.num_envs, self.obs_dim)

        # Save observation to buffer
        self._save_obs(obs)

        return self._obs_from_buf()

    def step(self, actions) -> VecEnvStepReturn:
        """
        Returns:
        --------
            torch.Tensor (max_agent_count * num_worlds, obs_dim): Observations.
            torch.Tensor (max_agent_count * num_worlds): Rewards.
            torch.Tensor (max_agent_count * num_worlds): Dones.
            torch.Tensor (max_agent_count * num_worlds, info_dim): Information.

        Note:
        -------
            In multi-agent settings some agents may be done before others.
            To handle this, we return done is 1 at the first time step the
            agent is done. After that, we return nan for the rewards, infos
            and done for that agent until the end of the episode.
        """

        # Unsqueeze action tensor to a shape the gpudrive env expects
        actions = actions.reshape((self.num_worlds, self.max_agent_count))

        # Step the environment
        obs, reward, done, info = self._env.step(actions)

        # Storage: Fill buffer with nan values
        self.buf_rews = torch.full(
            (self.num_worlds, self.max_agent_count), fill_value=float("nan")
        ).to(self.device)
        self.buf_dones = torch.full(
            (self.num_worlds, self.max_agent_count), fill_value=float("nan")
        ).to(self.device)
        buf_obs = torch.full(
            (self.num_worlds, self.max_agent_count, self.obs_dim),
            fill_value=float("nan"),
        ).to(self.device)

        # Override nan placeholders for alive agents
        self.buf_rews[~self.dead_agent_mask] = reward[~self.dead_agent_mask]
        self.buf_dones[~self.dead_agent_mask] = done[~self.dead_agent_mask].to(
            torch.float32
        )
        buf_obs[~self.dead_agent_mask] = obs[~self.dead_agent_mask]

        # Flatten over num_worlds and max_agent_count and store
        obs = obs.reshape(self.num_envs, self.obs_dim)
        self._save_obs(obs)

        # Reset episode if all agents in all worlds are done before
        # the end of the episode
        if (
            done[self.controlled_agent_mask].sum().item()
            == self._tot_controlled_valid_agents
        ):
            # Update infos
            self._update_info_dict(info)
            self.num_episodes += 1

            # Reset environment
            obs = self.reset()

        else:
            # Update dead agent mask: Set to True if agent is done before
            # the end of the episode
            self.dead_agent_mask = torch.logical_or(self.dead_agent_mask, done)

        return (
            self._obs_from_buf(),
            torch.clone(self.buf_rews).reshape(self.num_envs),
            torch.clone(self.buf_dones).reshape(self.num_envs),
            torch.clone(info).reshape(self.num_envs, self.info_dim),
        )

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

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

    def _update_info_dict(self, info) -> None:
        """Update the info logger."""

        controlled_agent_info = info[self.controlled_agent_mask]

        self.infos["off_road"] += controlled_agent_info[:, 0].sum().item()
        self.infos["veh_collisions"] += (
            controlled_agent_info[:, 1].sum().item()
        )
        self.infos["non_veh_collision"] += (
            controlled_agent_info[:, 2].sum().item()
        )
        self.infos["goal_achieved"] += controlled_agent_info[:, 3].sum().item()

    def _reset_rollout_loggers(self) -> None:
        self.num_episodes = 0
        self.infos = {
            "off_road": 0,
            "veh_collisions": 0,
            "non_veh_collision": 0,
            "goal_achieved": 0,
        }

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(
        self, method_name, *method_args, indices=None, **method_kwargs
    ):
        if "method" == "render":
            return self._env.render()
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()

    def get_images(self, policy=None) -> Sequence[Optional[np.ndarray]]:
        frames = [self._env.render()]
        return frames

    @property
    def _tot_controlled_valid_agents(self):
        return self._env.num_valid_controlled_agents


if __name__ == "__main__":

    config = EnvConfig()

    # Make environment
    env = SB3MultiAgentEnv(
        config=config,
        num_worlds=1,
        max_cont_agents=10,
        data_dir="formatted_json_v2_no_tl_train",
        device="cpu",
    )

    obs = env.reset()
    for global_step in range(200):

        print(f"Step: {90 - env._env.steps_remaining[0, 0, 0].item()}")

        # Random actions
        actions = torch.randint(0, env.action_space.n, (env.num_envs,))

        # Step
        obs, rew, done, info = env.step(actions)

        print(f"(out step) done: {done} \n")

        print(
            f"obs: {obs.shape} | rew: {rew.shape} | done: {done.shape} | info: {info.shape} \n"
        )
