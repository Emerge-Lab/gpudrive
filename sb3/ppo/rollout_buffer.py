import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), device=self.device,
                                     dtype=th.float32)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), 
                                device=self.device,
                                dtype=th.float32)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), 
                                device=self.device, dtype=th.float32)
        self.returns = th.zeros((self.buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.values = th.zeros((self.buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.valids = th.zeros((self.buffer_size, self.n_envs), device=self.device, dtype=th.bool)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: th.Tensor, valids: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = valids * (1.0 - dones.float())
                next_values = last_values
            else:
                next_non_terminal = self.valids[step + 1] * (1.0 - self.episode_starts[step + 1])
                next_values = self.values[step + 1]
            # TODO(ev) this might be accumulating reward across invalid states
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
        valid: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob
        self.valids[self.pos] = valid
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None, advantage_filter: float = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns", "valids"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # filter out low advantages but only on valid samples
        valid_indices = th.nonzero(self.valids.flatten()).squeeze()
        # Step 2: Select the corresponding values from advantages
        valid_advs = self.advantages.flatten()[valid_indices]
        total_num = valid_advs.shape[0]
        # Step 3: Compute k-th top value of the absolute values of advantages
        top_k_value = th.topk(th.abs(valid_advs), k=int(total_num * (1 - advantage_filter))).values[-1]
        # Step 4: Compare the absolute values of the selected advantages with k-th top value
        condition_met = th.abs(valid_advs) > top_k_value
        # Step 5: Return the valid indices that meet the above criteria
        indices = valid_indices[condition_met]
        assert indices.shape[0] > 0, "No valid advantage values found? Seems like a bug."

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < indices.shape[0]:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        assert th.all(self.valids[batch_inds].flatten()), "Invalid indices in batch"
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))