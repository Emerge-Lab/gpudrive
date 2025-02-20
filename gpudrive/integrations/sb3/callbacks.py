from collections import deque
import os
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from time import perf_counter
from gpudrive.visualize.utils import img_from_fig


class MultiAgentCallback(BaseCallback):
    """Stable Baselines3 callback for multi-agent gpudrive env."""

    def __init__(self, config, wandb_run=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.wandb_run = wandb_run
        self.num_rollouts = 0
        self.step_counter = 0
        self.policy_base_path = os.path.join(wandb.run.dir, "policies")
        os.makedirs(self.policy_base_path, exist_ok=True)
        self.perc_goal_achieved = deque(
            maxlen=config.logging_collection_window
        )
        self.perc_off_road = deque(maxlen=config.logging_collection_window)
        self.perc_veh_collisions = deque(
            maxlen=config.logging_collection_window
        )
        self.perc_non_veh_collision = deque(
            maxlen=config.logging_collection_window
        )
        self.num_agent_rollouts = deque(
            maxlen=config.logging_collection_window
        )
        self.perc_truncated = deque(maxlen=config.logging_collection_window)
        self.max_obs = deque(maxlen=config.logging_collection_window)
        self.min_obs = deque(maxlen=config.logging_collection_window)

        self._define_wandb_metrics()

    def _define_wandb_metrics(self):
        """Automatically set correct x-axis for metrics."""
        wandb.define_metric("global_step")
        metrics = [
            "metrics/mean_episode_reward_per_agent",
            "metrics/perc_goal_achieved",
            "metrics/perc_off_road",
            "metrics/perc_veh_collisions",
            "metrics/perc_non_veh_collision",
            "charts/obs_max",
            "charts/obs_min",
        ]
        for metric in metrics:
            wandb.define_metric(metric, step_metric="global_step")

    def _on_training_start(self) -> None:
        """This method is called before the first rollout starts."""
        self.start_training = perf_counter()
        self.log_first_to_95 = True

    def _on_training_end(self) -> None:
        """This method is called at the end of training."""
        if self.config.save_policy:
            self._save_policy_checkpoint()

    def _on_rollout_start(self) -> None:
        """Triggered before collecting new samples."""
        pass

    def _on_step(self) -> bool:
        """Will be called by the model after each call to `env.step()`."""
        self.step_counter += 1
        env_info = self.locals["env"].info_dict

        if env_info:
            self.num_agent_rollouts.append(env_info["num_controlled_agents"])
            self.perc_off_road.append(env_info["off_road"])
            self.perc_veh_collisions.append(env_info["veh_collisions"])
            self.perc_non_veh_collision.append(env_info["non_veh_collision"])
            self.perc_goal_achieved.append(env_info["goal_achieved"])
            self.perc_truncated.append(env_info["truncated"])
            self.max_obs.append(self.locals["env"].obs_alive.max().item())
            self.min_obs.append(self.locals["env"].obs_alive.min().item())

            if self.step_counter % self.config.log_freq == 0:
                self._log_metrics()
                self._log_obs_stats()

            if self.config.track_time_to_solve:
                self._log_time_to_solve()

    def _log_metrics(self):
        """Log performance metrics to wandb."""
        total_agents = sum(self.num_agent_rollouts)
        metrics = {
            "global_step": self.num_timesteps,
            "metrics/wallclock_time (s)": perf_counter() - self.start_training,
            "metrics/perc_off_road": sum(self.perc_off_road) / total_agents,
            "metrics/perc_veh_collisions": sum(self.perc_veh_collisions)
            / total_agents,
            "metrics/perc_non_veh_collision": sum(self.perc_non_veh_collision)
            / total_agents,
            "metrics/perc_goal_achieved": sum(self.perc_goal_achieved)
            / total_agents,
            "metrics/perc_truncated": sum(self.perc_truncated) / total_agents,
        }
        wandb.log(metrics)

    def _log_obs_stats(self):
        """Log observation statistics to wandb."""
        wandb.log(
            {
                "charts/obs_max": np.array(self.max_obs).max(),
                "charts/obs_min": np.array(self.min_obs).min(),
            }
        )

    def _log_time_to_solve(self):
        """Log the time and steps taken to achieve 95% goal achievement."""
        if (
            sum(self.perc_goal_achieved) / sum(self.num_agent_rollouts) >= 0.95
            and self.log_first_to_95
        ):
            wandb.log(
                {
                    "charts/time_to_95": perf_counter() - self.start_training,
                    "charts/steps_to_95": self.num_timesteps,
                }
            )
            self.log_first_to_95 = False

    def _on_rollout_end(self) -> None:
        """Triggered before updating the policy."""
        rewards = (
            torch.nan_to_num(self.locals["rollout_buffer"].rewards, nan=0)
            .sum()
            .item()
        )
        completions = torch.nan_to_num(
            self.locals["rollout_buffer"].episode_starts
        ).sum()

        if (
            self.config.save_policy
            and self.num_rollouts % self.config.save_policy_freq == 0
        ):
            self._save_policy_checkpoint()

        self.num_rollouts += 1
        wandb.log(
            {
                "global_step": self.num_timesteps,
                "metrics/mean_episode_reward_per_agent": rewards / completions,
            }
        )

    def _save_policy_checkpoint(self) -> None:
        """Save the policy locally and to wandb."""
        path = os.path.join(
            self.policy_base_path, f"policy_{self.num_timesteps}.zip"
        )
        self.model.save(path)
        if self.wandb_run is not None:
            wandb.save(path, base_path=self.policy_base_path)
        print(f"Saved policy on step {self.num_timesteps:,} at: {path}")
