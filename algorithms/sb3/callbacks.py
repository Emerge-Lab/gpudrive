from collections import deque

import os
import logging
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from time import perf_counter

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
        self.worst_to_best_scene_perf_idx = None

        # TODO(ev) don't just define these here
        self.mean_ep_reward_per_agent = deque(
            maxlen=self.config.logging_collection_window
        )
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
        self.mean_reward_per_episode = deque(
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
        wandb.define_metric("charts/max_obs", step_metric="global_step")
        wandb.define_metric("charts/min_obs", step_metric="global_step")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start_training = perf_counter()
        self.log_first_to_95 = True
        self.log_first_to_90 = True

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

        self.step_counter += 1
        if len(self.locals["env"].info_dict) > 0:
            # total number of agents
            self.num_agent_rollouts.append(
                self.locals["env"].info_dict["num_finished_agents"]
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
            self.mean_reward_per_episode.append(
                self.locals["env"].info_dict["mean_reward_per_episode"]
            )
            self.perc_truncated.append(
                self.locals["env"].info_dict["truncated"]
            )
            self.max_obs.append(self.locals["obs_tensor"].max().item())
            self.min_obs.append(self.locals["obs_tensor"].min().item())
            

            if self.step_counter % self.config.log_freq == 0:
                      
                wandb.log(
                    {
                        "metrics/mean_ep_reward_per_agent": sum(
                            self.mean_reward_per_episode
                        )
                        / sum(self.num_agent_rollouts),
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
                        "charts/max_obs": np.array(self.max_obs).max(),
                        "charts/min_obs": np.array(self.min_obs).min(),
                    }
                )
            
            if self.config.track_time_to_solve:
                if sum(self.perc_goal_achieved) / sum(self.num_agent_rollouts) >= 0.9 and self.log_first_to_90:
                    wandb.log({
                        'charts/time_to_90': perf_counter() - self.start_training,
                        'charts/steps_to_90': self.num_timesteps,
                    })
                    self.log_first_to_90 = False
                
                if sum(self.perc_goal_achieved) / sum(self.num_agent_rollouts) >= 0.95 and self.log_first_to_95:
                    wandb.log({
                        'charts/time_to_95': perf_counter() - self.start_training,
                        'charts/steps_to_95': self.num_timesteps,
                    })
                    self.log_first_to_95 = False
                
            
            # LOG FAILURE MODES AND DISTRIBUTIONS
            if self.locals["env"].log_agg_world_info:

                agg_world_info_dict = self.locals["env"].aggregate_world_dict

                goal_achieved_dist = [
                    agg_world_info_dict[i][1].item()
                    for i in range(len(agg_world_info_dict))
                ]
                off_road_dist = [
                    agg_world_info_dict[i][2].item()
                    for i in range(len(agg_world_info_dict))
                ]
                veh_coll_dist = [
                    agg_world_info_dict[i][3].item()
                    for i in range(len(agg_world_info_dict))
                ]

                # Sort indices from worst to best performance
                self.worst_to_best_scene_perf_idx = np.argsort(
                    -np.array(veh_coll_dist)
                )

                self.best_to_worst_scene_perf_idx = np.argsort(
                    np.array(veh_coll_dist)
                )

                # Log as histograms to wandb
                wandb.log(
                    {
                        "charts/goal_achieved_dist": wandb.Histogram(
                            goal_achieved_dist
                        )
                    }
                )
                wandb.log(
                    {"charts/off_road_dist": wandb.Histogram(off_road_dist)}
                )
                wandb.log(
                    {"charts/veh_coll_dist": wandb.Histogram(veh_coll_dist)}
                )

                # Render
                if self.config.render:
                    # LOG FAILURE MODES
                    if (
                        self.config.log_failure_modes_after is None
                        or self.num_timesteps
                        > self.config.log_failure_modes_after
                    ):
                        if self.num_rollouts % self.config.render_freq == 0:
                            if self.worst_to_best_scene_perf_idx is not None:
                                for (
                                    world_idx
                                ) in self.worst_to_best_scene_perf_idx[
                                    : self.config.render_n_worlds
                                ]:
                                    controlled_agents_in_world = (
                                        self.locals["env"]
                                        .controlled_agent_mask[world_idx]
                                        .sum()
                                        .item()
                                    )

                                    self._create_and_log_video(
                                        render_world_idx=world_idx,
                                        caption=f"Index: {world_idx} | Cont. agents: {controlled_agents_in_world} | OR: {off_road_dist[world_idx]:.2f} | CR: {veh_coll_dist[world_idx]:.2f}",
                                        render_type="failure_modes",
                                    )

                    # LOG BEST SCENES
                    if (
                        self.config.log_success_modes_after is None
                        or self.num_timesteps
                        > self.config.log_success_modes_after
                    ):
                        if self.num_rollouts % self.config.render_freq == 0:
                            if self.best_to_worst_scene_perf_idx is not None:

                                for (
                                    world_idx
                                ) in self.best_to_worst_scene_perf_idx[
                                    : self.config.render_n_worlds
                                ]:
                                    controlled_agents_in_world = (
                                        self.locals["env"]
                                        .controlled_agent_mask[world_idx]
                                        .sum()
                                        .item()
                                    )

                                    goal_achieved = goal_achieved_dist[
                                        world_idx
                                    ]
                                    total_collision_rate = (
                                        veh_coll_dist[world_idx]
                                        + off_road_dist[world_idx]
                                    )

                                    self._create_and_log_video(
                                        render_world_idx=world_idx,
                                        caption=f"Index: {world_idx} | Cont. agents: {controlled_agents_in_world} | GR: {goal_achieved:.2f} | OR + CR: {total_collision_rate:.2f}",
                                        render_type="best_scenes",
                                    )

                # Reset
                self.locals["env"].log_agg_world_info = False
                self.locals["env"].aggregate_world_dict = {}

    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy.
        """

        # Model checkpointing
        if self.config.save_policy:
            if self.num_rollouts % self.config.save_policy_freq == 0:
                self._save_policy_checkpoint()

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

    def _create_and_log_video(
        self,
        render_world_idx=0,
        caption=" ",
        render_type=" ",
        sub_directory="",
    ):
        """Make a video and log to wandb.
        Note: Currently only works a single world."""
        policy = self.model
        env = self.locals["env"]

        obs = env.reset()

        frames = []

        for _ in range(90):

            action, _ = policy.predict(obs.detach().cpu().numpy())
            action = torch.Tensor(action).to("cuda")

            # Step the environment
            obs, _, _, _ = env.step(action)

            frame = env._env.render(world_render_idx=render_world_idx)
            frames.append(frame)

        frames = np.array(frames)

        wandb.log(
            {
                f"{render_type} | Global step: {self.num_timesteps:,}": wandb.Video(
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
            f"Saved policy on global_step {self.num_timesteps:,} at: \n {self.path}"
        )
