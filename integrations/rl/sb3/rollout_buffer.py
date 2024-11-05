"""Module containing regularized PPO algorithm."""
import logging
from typing import Generator, Optional
import gymnasium as gym
import numpy as np
import torch
from typing import Union, NamedTuple
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer

logging.getLogger(__name__)


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class MaskedRolloutBuffer(BaseBuffer):
    """Custom SB3 RolloutBuffer class that filters out invalid samples."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "auto",
        storage_device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.storage_device = storage_device
        self.reset()

    def reset(self) -> None:
        """Reset the buffer."""
        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.returns = torch.zeros(
            (self.buffer_size, self.n_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.episode_starts = torch.zeros(
            (self.buffer_size, self.n_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.values = torch.zeros(
            (self.buffer_size, self.n_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.advantages = torch.zeros(
            (self.buffer_size, self.n_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.generator_ready = False
        super().reset()

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        EDIT: We do rollouts on the GPU --> convert torch arrays to torch tensors
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs.to(self.storage_device)
        self.actions[self.pos] = action.to(self.storage_device)
        self.rewards[self.pos] = reward.to(self.storage_device)
        self.episode_starts[self.pos] = episode_start.to(self.storage_device)
        self.values[self.pos] = value.flatten().to(self.storage_device)
        self.log_probs[self.pos] = log_prob.clone().to(self.storage_device)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: torch.Tensor
    ) -> None:
        """GAE (General Advantage Estimation) to compute advantages and returns."""
        # Convert to numpy
        last_values = last_values.clone().flatten().to(self.storage_device)
        dones = dones.clone().flatten().to(self.storage_device)

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # EDIT_1: Map NaNs to 1
                dones = torch.nan_to_num(dones, nan=1.0)

                next_non_terminal = 1.0 - dones
                next_values = last_values

            else:
                # EDIT_1: Map NaNs to 1
                episode_starts = torch.nan_to_num(
                    self.episode_starts[step + 1], nan=1.0
                )

                next_non_terminal = 1.0 - episode_starts
                next_values = self.values[step + 1]

            delta = (
                torch.nan_to_num(
                    self.rewards[step], nan=0
                )  # EDIT_2: Set invalid rewards to zero
                + torch.nan_to_num(
                    self.gamma * next_values * next_non_terminal, nan=0
                )  # EDIT_3: Set invalid rewards to zero
                - torch.nan_to_num(
                    self.values[step], nan=0
                )  # EDIT_4: Set invalid values to zero
            )

            last_gae_lam = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

        assert not torch.isnan(
            self.advantages
        ).any(), "Advantages arr contains NaN values: Check GAE computation"

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
            ]
            # Create mask
            self.valid_samples_mask = ~torch.isnan(
                self.swap_and_flatten(self.__dict__["rewards"])
            )

            # Flatten data
            # EDIT_5: And mask out invalid samples
            for tensor in _tensor_names:
                if tensor == "observations":
                    self.__dict__[tensor] = self.swap_and_flatten(
                        self.__dict__[tensor]
                    )[self.valid_samples_mask.flatten(), :]
                else:
                    self.__dict__[tensor] = self.swap_and_flatten(
                        self.__dict__[tensor]
                    )[self.valid_samples_mask]

                assert not torch.isnan(
                    self.__dict__[tensor]
                ).any(), f"{tensor} tensor contains NaN values; something went wrong"

            self.generator_ready = True

        # EDIT_6: Compute total number of samples and create indices
        total_num_samples = self.valid_samples_mask.sum()
        indices = torch.randperm(total_num_samples)

        # if self.__dict__["observations"].max() > 1 or self.__dict__["observations"].min() < -1:
        #     print("Observations are out of range")

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = total_num_samples

        start_idx = 0
        while start_idx < total_num_samples:
            yield self._get_samples(
                indices[start_idx : start_idx + batch_size]
            )
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch]
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
