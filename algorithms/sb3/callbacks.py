from collections import deque

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
        self.num_rollouts = 0

        # TODO(ev) don't just define these here
        self.mean_ep_reward_per_agent = deque(maxlen=self.config.logging_collection_window)
        self.perc_goal_achieved = deque(maxlen=self.config.logging_collection_window)
        self.perc_off_road = deque(maxlen=self.config.logging_collection_window)
        self.perc_veh_collisions = deque(maxlen=self.config.logging_collection_window)
        self.perc_non_veh_collision = deque(maxlen=self.config.logging_collection_window)
        self.num_agent_rollouts = deque(maxlen=self.config.logging_collection_window)
        self.mean_reward_per_episode = deque(maxlen=self.config.logging_collection_window)
        self.perc_truncated = deque(maxlen=self.config.logging_collection_window)
        
        self._define_wandb_metrics()  # Set x-axis for metrics

    def _define_wandb_metrics(self):
        """Automatically set correct x-axis for metrics."""
        wandb.define_metric("global_step")
        wandb.define_metric(
            "metrics/mean_ep_reward_per_agent", step_metric="global_step"
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

    def unbatchify(self, tensor):
        """Unsqueeze flattened tensor to (num_worlds, max_num_objects, ...) shape."""
        return tensor.reshape(
            self.locals["env"].num_worlds,
            self.locals["env"].max_agent_count,
            -1,
        )

    def _on_step(self) -> bool:
        """
        Will be called by the model after each call to `env.step()`.
        """

        # LOG AND RESET METRICS AFTER EACH EPISODE (when all worlds are done)
        if len(self.locals["env"].info_dict) > 0:
            # total number of agents
            self.num_agent_rollouts.append(self.locals["env"].info_dict[
                "num_finished_agents"
            ])
            self.perc_off_road.append(self.locals["env"].info_dict["off_road"])
            self.perc_veh_collisions.append(self.locals["env"].info_dict["veh_collisions"])
            self.perc_non_veh_collision.append(self.locals["env"].info_dict["non_veh_collision"])
            self.perc_goal_achieved.append(self.locals["env"].info_dict["goal_achieved"])
            self.mean_reward_per_episode.append(self.locals["env"].info_dict["mean_reward_per_episode"])
            self.perc_truncated.append(self.locals["env"].info_dict["truncated"])
            # TODO(ev) add logging of agents that did not achieve their goals
            wandb.log(
                {
                    "global_step": self.num_timesteps,
                    # TODO(ev) this metric is broken
                    "metrics/mean_ep_reward_per_agent": sum(self.mean_reward_per_episode) / sum(self.num_agent_rollouts),
                    "metrics/perc_off_road": (
                        sum(self.perc_off_road) / sum(self.num_agent_rollouts)
                    )
                    * 100,
                    "metrics/perc_veh_collisions": (
                        sum(self.perc_veh_collisions) / sum(self.num_agent_rollouts)
                    )
                    * 100,
                    "metrics/perc_non_veh_collision": (
                        sum(self.perc_non_veh_collision) / sum(self.num_agent_rollouts)
                    )
                    * 100,
                    "metrics/perc_goal_achieved": (
                        sum(self.perc_goal_achieved) / sum(self.num_agent_rollouts)
                    )
                    * 100,
                    "metrics/perc_truncated": (
                        sum(self.perc_truncated) / sum(self.num_agent_rollouts)
                    )
                    * 100,
                }
            )

    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy.
        """

        # Render the environment
        if self.config.render:
            if self.num_rollouts % self.config.render_freq == 0:
                for world_idx in range(self.config.render_n_worlds):
                    self._create_and_log_video(render_world_idx=world_idx)

        self.num_rollouts += 1

    def _batchify_and_filter_obs(self, obs, env, render_world_idx=0):
        # Unsqueeze
        obs = obs.reshape((env.num_worlds, env.max_agent_count, -1))

        # Only select obs for the render env
        obs = obs[render_world_idx, :, :]

        return obs[env.controlled_agent_mask[render_world_idx, :]]

    def _pad_actions(self, pred_actions, env, render_world_idx):
        """Currently we're only rendering the 0th world index."""

        actions = torch.full(
            (env.num_worlds, env.max_agent_count), fill_value=float("nan")
        ).to("cpu")

        world_cont_agent_mask = env.controlled_agent_mask[
            render_world_idx, :
        ].to("cpu")

        actions[render_world_idx, :][world_cont_agent_mask] = torch.Tensor(
            pred_actions
        ).to("cpu")
        return actions

    def _create_and_log_video(self, render_world_idx=0):
        """Make a video and log to wandb.
        Note: Currently only works a single world."""
        policy = self.model
        env = self.locals["env"]

        obs = env.reset()
        obs = self._batchify_and_filter_obs(obs, env)

        frames = []

        for _ in range(90):

            action, _ = policy.predict(obs.detach().cpu().numpy())
            action = self._pad_actions(action, env, render_world_idx)

            # Step the environment
            obs, _, _, _ = env.step(action)
            obs = self._batchify_and_filter_obs(obs, env)

            frame = env.render()
            frames.append(frame)

        frames = np.array(frames)

        wandb.log(
            {
                f"video_{render_world_idx}": wandb.Video(
                    np.moveaxis(frames, -1, 1),
                    fps=10,
                    format="gif",
                    caption=f"Global step: {self.num_timesteps:,}",
                )
            }
        )
