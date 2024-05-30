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
        self.num_agent_rollouts = 0 # This tracks total agent rollouts
        
        # TODO(ev) don't just define these here
        self.mean_ep_reward_per_agent = 0
        self.perc_goal_achieved = 0
        self.perc_off_road = 0
        self.perc_veh_collisions = 0
        self.perc_non_veh_collision = 0 

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
            total_valid_agents = self.locals["env"].info_dict["num_finished_agents"]

            self.perc_off_road += self.locals["env"].info_dict["off_road"]
            self.perc_veh_collisions += self.locals["env"].info_dict["veh_collisions"]
            self.perc_non_veh_collision += self.locals["env"].info_dict["non_veh_collision"]
            self.perc_goal_achieved += self.locals["env"].info_dict["goal_achieved"]
            self.num_agent_rollouts += total_valid_agents
            wandb.log(
                {
                    "global_step": self.num_timesteps,
                    # TODO(ev) this metric is broken
                    "metrics/mean_ep_reward_per_agent": self.locals[
                        "env"
                    ].tot_reward_per_episode
                    / total_valid_agents,
                    "metrics/perc_off_road": (
                        self.perc_off_road / self.num_agent_rollouts
                    )
                    * 100,
                    "metrics/perc_veh_collisions": (
                        self.perc_veh_collisions / self.num_agent_rollouts
                    )
                    * 100,
                    "metrics/perc_non_veh_collision": (
                        self.perc_non_veh_collision / self.num_agent_rollouts
                    )
                    * 100,
                    "metrics/perc_goal_achieved": (
                        self.perc_goal_achieved / self.num_agent_rollouts
                    )
                    * 100,
                }
            )

            # TODO (dc): Works, valid but hacky way to reset metrics
            # The tricky thing is that the env resets when done (in step), and the callback
            # call is after step
            # We use a reset flag to reset the env wrapper metrics
            # TODO(ev) this is broken
            self.locals["env"].tot_reward_per_episode = 0
            # self.locals["env"].info_dict = {
            #     "off_road": 0,
            #     "veh_collisions": 0,
            #     "non_veh_collision": 0,
            #     "goal_achieved": 0,
            # }

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
