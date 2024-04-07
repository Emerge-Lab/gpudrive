import logging
import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy


class MultiAgentCallback(BaseCallback):
    """SB3 callback for gpudrive."""

    def __init__(
        self,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.wandb_run = wandb_run

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        rewards = np.nan_to_num(
            (
                self.locals["rollout_buffer"]
                .rewards.cpu()
                .detach()
                .numpy()
                .flatten()
            ),
            nan=0,
        )

        observations = (
            self.locals["rollout_buffer"].observations.cpu().detach().numpy()
        )

        num_episodes_in_rollout = self.locals[
            "rollout_buffer"
        ].episode_starts.sum()

        self.logger.record("rollout/global_step", self.num_timesteps)
        self.logger.record(
            "rollout/num_episodes_in_rollout", num_episodes_in_rollout.item()
        )
        self.logger.record("rollout/sum_reward", rewards.sum())
        self.logger.record(
            "rollout/avg_reward",
            (rewards.sum() / num_episodes_in_rollout).item(),
        )

        self.logger.record("rollout/obs_max", observations.max())
        self.logger.record("rollout/obs_min", observations.min())
