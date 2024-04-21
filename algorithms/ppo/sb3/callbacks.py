import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class MultiAgentCallback(BaseCallback):
    """SB3 callback for gpudrive."""

    def __init__(
        self,
        config,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
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

        # TODO: fix this, currently only works when we have a fixed number of agents
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
        # for agent_idx in range(num_controlled_agents):
        #     self.logger.record(
        #         f"rollout/avg_agent_rew{agent_idx}",
        #         rewards[:, agent_idx].sum() / num_episodes_in_rollout,
        #     )

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

        # Render the environment
        if self.config.render:
            self._create_and_log_video()

    def _create_and_log_video(self):
        """Make a video and log to wandb.
        Note: Currently only works a single world."""
        policy = self.model
        env = self.locals["env"]

        obs = env.reset()
        frames = []

        for _ in range(90):

            action, _ = policy.predict(obs.detach().cpu().numpy())

            # Step the environment
            obs, reward, done, info = env.step(action)

            frame = env.render()
            frames.append(frame.T)

        frames = np.array(frames)

        wandb.log({"video": wandb.Video(frames, fps=5, format="gif")})
