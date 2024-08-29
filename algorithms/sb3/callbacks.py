from collections import deque

import os
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from time import perf_counter

EPISODE_LENGTH = 91


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
        self.num_rollouts = 0
        self.step_counter = 0
        self.policy_base_path = os.path.join(wandb.run.dir, "policies")
        if self.policy_base_path is not None:
            os.makedirs(self.policy_base_path, exist_ok=True)
        self.perc_goal_achieved = deque(
            maxlen=self.config.logging_collection_window
        )
        self.perc_off_road = deque(
            maxlen=self.config.logging_collection_window
        )
        self.perc_veh_collisions = deque(
            maxlen=self.config.logging_collection_window
        )
        self.perc_non_veh_collision = deque(
            maxlen=self.config.logging_collection_window
        )
        self.num_agent_rollouts = deque(
            maxlen=self.config.logging_collection_window
        )
        self.perc_truncated = deque(
            maxlen=self.config.logging_collection_window
        )
        self.max_obs = deque(maxlen=self.config.logging_collection_window)
        self.min_obs = deque(maxlen=self.config.logging_collection_window)

        self._define_wandb_metrics()  # Set x-axis for metrics

    def _define_wandb_metrics(self):
        """Automatically set correct x-axis for metrics."""
        wandb.define_metric("global_step")
        wandb.define_metric(
            "metrics/mean_episode_reward_per_agent", step_metric="global_step"
        )
        wandb.define_metric(
            "metrics/perc_goal_achieved", step_metric="global_step"
        )
        wandb.define_metric("metrics/perc_off_road", step_metric="global_step")
        wandb.define_metric(
            "metrics/perc_veh_collisions", step_metric="global_step"
        )
        wandb.define_metric(
            "metrics/perc_non_veh_collision", step_metric="global_step"
        )
        wandb.define_metric("charts/obs_max", step_metric="global_step")
        wandb.define_metric("charts/obs_min", step_metric="global_step")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start_training = perf_counter()
        self.log_first_to_95 = True

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        """
        # Save the policy before ending the run
        if self.config.save_policy and self.policy_base_path is not None:
            self._save_policy_checkpoint()

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

        self.step_counter += 1
        if len(self.locals["env"].info_dict) > 0:
            # total number of agents
            self.num_agent_rollouts.append(
                self.locals["env"].info_dict["num_controlled_agents"]
            )
            self.perc_off_road.append(self.locals["env"].info_dict["off_road"])
            self.perc_veh_collisions.append(
                self.locals["env"].info_dict["veh_collisions"]
            )
            self.perc_non_veh_collision.append(
                self.locals["env"].info_dict["non_veh_collision"]
            )
            self.perc_goal_achieved.append(
                self.locals["env"].info_dict["goal_achieved"]
            )
            self.perc_truncated.append(
                self.locals["env"].info_dict["truncated"]
            )
            
            self.max_obs.append(self.locals["env"].obs_alive.max().item())
            self.min_obs.append(self.locals["env"].obs_alive.min().item())

            if self.step_counter % self.config.log_freq == 0:
                
                wandb.log(
                    {
                        "metrics/wallclock_time (s)": perf_counter()
                        - self.start_training,
                        "metrics/global_step": self.num_timesteps,
                        "metrics/perc_off_road": (
                            sum(self.perc_off_road)
                            / sum(self.num_agent_rollouts)
                        )
                        * 100,
                        "metrics/perc_veh_collisions": (
                            sum(self.perc_veh_collisions)
                            / sum(self.num_agent_rollouts)
                        )
                        * 100,
                        "metrics/perc_non_veh_collision": (
                            sum(self.perc_non_veh_collision)
                            / sum(self.num_agent_rollouts)
                        )
                        * 100,
                        "metrics/perc_goal_achieved": (
                            sum(self.perc_goal_achieved)
                            / sum(self.num_agent_rollouts)
                        )
                        * 100,
                        "metrics/perc_truncated": (
                            sum(self.perc_truncated)
                            / sum(self.num_agent_rollouts)
                        )
                        * 100,
                    }
                )

                wandb.log(
                    {
                        "charts/obs_max": np.array(self.max_obs).max(),
                        "charts/obs_min": np.array(self.min_obs).min(),
                    }
                )

            if self.config.track_time_to_solve:

                if (
                    sum(self.perc_goal_achieved) / sum(self.num_agent_rollouts)
                    >= 0.95
                    and self.log_first_to_95
                ):
                    wandb.log(
                        {
                            "charts/time_to_95": perf_counter()
                            - self.start_training,
                            "charts/steps_to_95": self.num_timesteps,
                        }
                    )
                    self.log_first_to_95 = False

    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy.
        """
        
        # Get rewards, filter out NaNs from dead agents
        # This has shape: (num_steps, num_agents x num_worlds)
        rewards_across_worlds = torch.nan_to_num(self.locals["rollout_buffer"].rewards, nan=0).sum().item()  
        
        # Number of times an episode was completed, across all worlds and agents  
        num_completions_in_rollout = torch.nan_to_num(self.locals["rollout_buffer"].episode_starts).sum()

        # Log a number of videos
        if self.config.render:
            if self.num_rollouts % self.config.render_freq == 0:
                for world_idx in range(self.config.render_n_worlds):
                    self._create_and_log_video(
                        render_world_idx=world_idx,
                        video_title=f"Global step: {self.num_timesteps:,}",
                        caption=f"Env: {world_idx}",
                    )

        # Model checkpointing
        if self.config.save_policy:
            if self.num_rollouts % self.config.save_policy_freq == 0:
                self._save_policy_checkpoint()

        self.num_rollouts += 1
        wandb.log({
            "global_step": self.num_timesteps,
            "metrics/mean_episode_reward_per_agent": rewards_across_worlds / num_completions_in_rollout,
        })

    def _create_and_log_video(
        self,
        render_world_idx,
        video_title=" ",
        caption="" "",
    ):
        """Make a video and log to wandb."""
        policy = self.model
        base_env = self.locals["env"]._env
        action_tensor = torch.zeros(
            (base_env.num_worlds, base_env.max_agent_count)
        )

        obs = base_env.reset()

        control_mask = self.locals["env"].controlled_agent_mask.clone()

        # Flatten over num_worlds and max_agent_count
        obs = obs[control_mask].reshape(
            self.locals["env"].num_envs, self.locals["env"].obs_dim
        )

        frames = []

        for step_num in range(EPISODE_LENGTH):

            actions, _ = policy.predict(obs.detach().cpu().numpy())
            actions = torch.Tensor(actions)

            action_tensor[base_env.cont_agent_mask] = actions

            # Step the environment
            base_env.step_dynamics(action_tensor)

            # Get info
            obs = base_env.get_obs()
            obs = obs[control_mask].reshape(
                self.locals["env"].num_envs, self.locals["env"].obs_dim
            )

            done = base_env.get_dones()

            # Render
            if step_num % 2 == 0:
                frame = base_env.render(world_render_idx=render_world_idx)
                frames.append(frame)

            if done[control_mask].sum() == control_mask.sum():
                break

        # Log
        frames = np.array(frames)

        wandb.log(
            {
                f"{video_title}": wandb.Video(
                    np.moveaxis(frames, -1, 1),
                    fps=15,
                    format="gif",
                    caption=caption,
                )
            }
        )

    def _save_policy_checkpoint(self) -> None:
        """Save the policy locally and to wandb."""

        self.path = os.path.join(
            self.policy_base_path,
            f"policy_{self.num_timesteps}.zip",
        )
        self.model.save(self.path)
        if self.wandb_run is not None:
            wandb.save(self.path, base_path=self.policy_base_path)

        print(
            f"Saved policy on step {self.num_timesteps:,} at: \n {self.path}"
        )
