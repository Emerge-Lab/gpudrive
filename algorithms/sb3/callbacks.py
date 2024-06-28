import logging
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from time import perf_counter
from algorithms.sb3.wandb_wrapper import WindowedCounter

class MultiAgentCallback(BaseCallback):
    """SB3 callback for gpudrive."""

    def __init__(
        self,
        config,
        wandb_logger,
        policy_checkpointer,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.wandb = wandb_logger
        self.checkpointer = policy_checkpointer
        self.num_rollouts = 0
        self.step_counter = 0
        self.worst_to_best_scene_perf_idx = None

        # TODO(ev) don't just define these here
        self.wandb.define_metric("global_step", lambda: self.num_timesteps)
        self.num_agent_rollouts = WindowedCounter(self.config.logging_collection_window)
        
        self.mean_ep_reward_per_agent = WindowedCounter(self.config.logging_collection_window)
        self.wandb.define_metric("metrics/mean_ep_reward_per_agent", lambda: self.mean_reward_per_episode.value() / self.num_agent_rollouts.value())
        
        self.perc_goal_achieved = WindowedCounter(self.config.logging_collection_window)
        self.wandb.define_metric("metrics/perc_goal_achieved",
                               lambda: self.perc_goal_achieved.value() / self.num_agent_rollouts.value() * 100)
        
        self.perc_off_road = WindowedCounter(self.config.logging_collection_window)
        self.wandb.define_metric("metrics/perc_off_road",
                               lambda: self.perc_off_road.value() / self.num_agent_rollouts.value() * 100)
        
        self.perc_veh_collisions = WindowedCounter(self.config.logging_collection_window)
        self.wandb.define_metric("metrics/perc_veh_collisions",
                               lambda: self.perc_veh_collisions.value() / self.num_agent_rollouts.value() * 100)
        
        self.perc_non_veh_collision = WindowedCounter(self.config.logging_collection_window)
        self.wandb.define_metric("metrics/perc_non_veh_collision",
                               lambda: self.perc_non_veh_collision.value() / self.num_agent_rollouts.value() * 100)
        
        self.mean_reward_per_episode = WindowedCounter(self.config.logging_collection_window)

        self.perc_truncated = WindowedCounter(self.config.logging_collection_window)
        self.wandb.define_metric("metrics/perc_truncated", lambda: self.perc_truncated.value() / self.num_agent_rollouts.value() * 100)

        # self.max_obs = WindowedCounter(self.config.logging_collection_window, max)
        # self.wandb.define_metric("charts/obs_max", lambda: self.max_obs.value())
        
        # self.min_obs = WindowedCounter(self.config.logging_collection_window, min)
        # self.wandb.define_metric("charts/obs_min", lambda: self.min_obs.value())

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
        self.checkpointer.save(self.num_timesteps, self.model)

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
        if len(self.locals["env"].info_dict) <= 0:
            return True
        # total number of agents
        self.num_agent_rollouts.append(
            self.locals["env"].info_dict["num_finished_agents"].item()
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

        # self.max_obs.append(self.locals["obs_tensor"].max().item())
        # self.min_obs.append(self.locals["obs_tensor"].min().item())


        if self.step_counter % self.config.log_freq == 0:
            self.wandb.log_defined_metrics()

        if self.config.track_time_to_solve:
            if self.perc_goal_achieved / self.num_agent_rollouts >= 0.9 and self.log_first_to_90:
                self.wandb.log({
                    'charts/time_to_90': perf_counter() - self.start_training,
                    'charts/steps_to_90': self.num_timesteps,
                })
                self.log_first_to_90 = False
                
            if self.perc_goal_achieved / self.num_agent_rollouts >= 0.95 and self.log_first_to_95:
                self.wandb.log({
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

            self.wandb.log_histograms({"charts/goal_achieved_dist": goal_achieved_dist,
                                       "charts/off_road_dist": off_road_dist,
                                       "charts/veh_coll_dist": veh_coll_dist})

            # Render
            if self.config.render:

                # LOG FAILURE MODES
                if (
                    self.config.log_failure_modes_after is not None
                    and self.num_timesteps
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
                    self.config.log_success_modes_after is not None
                    and self.num_timesteps
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

        return True

    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy.
        """
        self.checkpointer.maybe_save(self.num_rollouts, self.num_timesteps, self.model)

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

        self.wandb.log_video(self.num_timesteps, frames=np.moveaxis(frames, -1, 1), fps=15, format="gif",  caption=caption)
