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

        # Get the total number of controlled agents we are controlling
        # The number of controllable agents is different per scenario
        num_episodes_in_rollout = self.locals["env"].num_episodes
        num_controlled_agents = self.locals["env"]._tot_controlled_valid_agents

        # Filter out all nans
        rollout_rewards = np.nan_to_num(
            (self.locals["rollout_buffer"].rewards.cpu().detach().numpy()),
            nan=0,
        )

        mean_reward_per_agent_per_episode = rollout_rewards.sum() / (
            num_episodes_in_rollout * num_controlled_agents
        )

        rollout_observations = (
            self.locals["rollout_buffer"].observations.cpu().detach().numpy()
        )

        # Average info across agents and episodes
        rollout_info = self.locals["env"].infos
        for key, value in rollout_info.items():
            self.locals["env"].infos[key] = value / (
                num_episodes_in_rollout * num_controlled_agents
            )
            self.logger.record(f"rollout/{key}", self.locals["env"].infos[key])

        # Log
        self.logger.record("rollout/global_step", self.num_timesteps)
        self.logger.record(
            "rollout/num_episodes_in_rollout",
            num_episodes_in_rollout,
        )
        self.logger.record("rollout/sum_reward", rollout_rewards.sum())
        self.logger.record(
            "rollout/avg_reward", mean_reward_per_agent_per_episode.item()
        )
        self.logger.record("rollout/obs_max", rollout_observations.max())
        self.logger.record("rollout/obs_min", rollout_observations.min())

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
            obs, _, _, _ = env.step(action)

            frame = env.render()
            frames.append(frame.T)

        frames = np.array(frames)

        wandb.log({"video": wandb.Video(frames, fps=5, format="gif")})
