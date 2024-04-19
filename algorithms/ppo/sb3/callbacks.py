import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
        Will be called by the model after each call to `env.step()`.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy.
        """

        # Filter out all nans
        rewards = np.nan_to_num(
            (self.locals["rollout_buffer"].rewards.cpu().detach().numpy()),
            nan=0,
        )

        # TODO: note that this only works when we have a fixed number of agents
        num_controlled_agents = rewards.shape[1]

        # Number of episodes in the rollout
        num_episodes_in_rollout = (
            np.nan_to_num(
                (
                    self.locals["rollout_buffer"]
                    .episode_starts.cpu()
                    .detach()
                    .numpy()
                ),
                nan=0,
            ).sum()
            / num_controlled_agents
        )

        # Rewards for each agent
        for agent_idx in range(num_controlled_agents):
            self.logger.record(
                f"rollout/avg_agent_rew{agent_idx}",
                rewards[:, agent_idx].sum() / num_episodes_in_rollout,
            )

        observations = (
            self.locals["rollout_buffer"].observations.cpu().detach().numpy()
        )

        num_episodes_in_rollout = np.nan_to_num(
            (
                self.locals["rollout_buffer"]
                .episode_starts.cpu()
                .detach()
                .numpy()
            ),
            nan=0,
        ).sum()

        self.logger.record("rollout/global_step", self.num_timesteps)
        self.logger.record(
            "rollout/num_episodes_in_rollout",
            num_episodes_in_rollout.item() / num_controlled_agents,
        )
        self.logger.record("rollout/sum_reward", rewards.sum())
        self.logger.record(
            "rollout/avg_reward",
            (rewards.sum() / (num_episodes_in_rollout)).item(),
        )

        self.logger.record("rollout/obs_max", observations.max())
        self.logger.record("rollout/obs_min", observations.min())

        # Get categorical max values
        self.logger.record("norm/speed_max", observations[:, :, 0].max())
        self.logger.record("norm/veh_len_max", observations[:, :, 1].max())
        self.logger.record("norm/veh_width_max", observations[:, :, 2].max())
        self.logger.record("norm/goal_coord_x", observations[:, :, 3].max())
        self.logger.record("norm/goal_coord_y", observations[:, :, 4].max())
        self.logger.record("norm/question_mark", observations[:, :, 5].max())
        self.logger.record("norm/L2_norm_to_goal", observations[:, :, 6].max())
