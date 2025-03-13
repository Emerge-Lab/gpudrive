"""Vectorized environment wrapper for multi-agent environments."""
import logging
import wandb
from typing import Optional, Sequence
import torch
import os
import gymnasium as gym
import random
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
)

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import RenderConfig
from gpudrive.visualize.utils import img_from_fig

logging.basicConfig(level=logging.INFO)


class SB3MultiAgentEnv(VecEnv):
    """Casts multi-agent environments as vectorized environments.

    Args:
    -----
        VecEnv (SB3 VecEnv): SB3 VecEnv base class.
    """

    def __init__(
        self,
        config,
        exp_config,
        max_cont_agents,
        device,
        render_mode="rgb_array",
        collision_weight=-.5,
        goal_achieved_weight=1,
        off_road_weight=-.5,        
        log_distance_weight=.01,
        render = False,
        render_3d=True,
        render_interval=2,
        render_k_scenarios=2,
        render_agent_obs=False,
        render_format="mp4",
        render_fps=15,
        zoom_radius=50,
        wandb_obj=None,
    ):
        #for rendering
        self.wandb_obj = wandb_obj
        self.render = render
        if self.render:
            assert self.wandb_obj != None
        self.render_interval = render_interval
        self.render_k_scenarios = render_k_scenarios
        self.render_agent_obs = render_agent_obs
        self.render_format = render_format
        self.render_fps = render_fps
        self.zoom_radius = zoom_radius
        self.iters = 0

        render_config = RenderConfig(
            render_3d=render_3d,
        )

        data_loader = SceneDataLoader(
            root=exp_config.data_dir,
            batch_size=exp_config.num_worlds,
            dataset_size=exp_config.resample_dataset_size,
            sample_with_replacement=exp_config.sample_with_replacement,
            shuffle=exp_config.shuffle_dataset,
        )

        self._env = GPUDriveTorchEnv(
            config=config,
            render_config=render_config,
            data_loader=data_loader,
            max_cont_agents=max_cont_agents,
            device=device,
        )
        self.config = config
        self.exp_config = exp_config
        self.all_scene_paths = [
            os.path.join(self.exp_config.data_dir, scene)
            for scene in sorted(os.listdir(self.exp_config.data_dir))
            if scene.startswith("tfrecord")
        ]
        self.unique_scene_paths = list(set(self.all_scene_paths))
        self.num_worlds = self._env.num_worlds
        self.max_agent_count = self._env.max_agent_count
        self.num_envs = self._env.cont_agent_mask.sum().item()
        self.device = device
        self.controlled_agent_mask = self._env.cont_agent_mask.clone()
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self._env.observation_space.shape, np.float32
        )

        self.obs_dim = self._env.observation_space.shape[-1]
        self.info_dim = 5
        self.render_mode = render_mode
        self.agent_step = torch.zeros(
            (self.num_worlds, self.max_agent_count)
        ).to(self.device)
        self.actions_tensor = torch.zeros(
            (self.num_worlds, self.max_agent_count)
        ).to(self.device)
        # Storage: Fill buffer with nan values
        self.buf_rews = torch.full(
            (self.num_worlds, self.max_agent_count), fill_value=float("nan")
        ).to(self.device)
        self.buf_dones = torch.full(
            (self.num_worlds, self.max_agent_count), fill_value=float("nan")
        ).to(self.device)
        self.buf_obs = torch.full(
            (self.num_envs, self.obs_dim),
            fill_value=float("nan"),
        ).to(self.device)

        self.num_episodes = 0

        self.collision_weight = collision_weight
        self.goal_achieved_weight = goal_achieved_weight
        self.off_road_weight = off_road_weight
        self.log_distance_weight = log_distance_weight

        # Setup rendering storage
        self.rendering_in_progress = {
            env_idx: False for env_idx in range(render_k_scenarios)
        }
        self.was_rendered_in_rollout = {
            env_idx: True for env_idx in range(render_k_scenarios)
        }
        self.frames = {env_idx: [] for env_idx in range(render_k_scenarios)}

    def _reset_seeds(self) -> None:
        """Reset all environments' seeds."""
        self._seeds = None

    def reset(self, world_idx=None, seed=None):
        """Reset environment and return initial observations.

        Returns:
        --------
            torch.Tensor (max_agent_count * num_worlds, obs_dim):
                Initial observation.
        """
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32)
        if world_idx is None:
            self._env.reset()
            obs = self._env.get_obs()

            # Make dead agent mask (True for dead or invalid agents)
            self.dead_agent_mask = ~self.controlled_agent_mask.clone()

            # Flatten over num_worlds and max_agent_count
            obs = obs[self.controlled_agent_mask].reshape(
                self.num_envs, self.obs_dim
            )

            return obs
        else:
            self._env.sim.reset(world_idx.item())

    def step(self, actions) -> VecEnvStepReturn:
        """
        Returns:
        --------
            torch.Tensor (max_agent_count * num_worlds, obs_dim): Next obs.
            torch.Tensor (max_agent_count * num_worlds): Rewards.
            torch.Tensor (max_agent_count * num_worlds): Dones.
            torch.Tensor (max_agent_count * num_worlds, info_dim): Info.

        Note:
        -------
            In multi-agent settings some agents may be done before others.
            To handle this, we return done is 1 at the first time step the
            agent is done. After that, we return nan for the rewards, infos
            and done for that agent until the end of the episode.
        """

        # Reset the info dict
        self.info_dict = {}

        # Unsqueeze action tensor to a shape the gpudrive env expects
        self.actions_tensor[self.controlled_agent_mask] = actions

        # Step the environment
        self._env.step_dynamics(self.actions_tensor)

        reward = self._env.get_rewards(collision_weight=self.collision_weight,
                                       goal_achieved_weight=self.goal_achieved_weight,
                                       off_road_weight=self.off_road_weight,
                                       log_distance_weight=self.log_distance_weight).clone()
        done = self._env.get_dones().clone()
        info = self._env.sim.info_tensor().to_torch()

        # CHECK IF A WORLD IS DONE -> RESET
        done_worlds = torch.where(
            (done.nan_to_num(0) * self.controlled_agent_mask).sum(dim=1)
            == self.controlled_agent_mask.sum(dim=1)
        )[0]
        self.render_env() if self.render else None
        if len(done_worlds) > 0:
            if self.render:
                for render_env_idx in range(self.render_k_scenarios):
                    self.log_video_to_wandb(render_env_idx, done_worlds)
            self._update_info_dict(info, done_worlds)
            self.num_episodes += len(done_worlds)
            self._env.sim.reset(done_worlds.tolist())
            self.episode_lengths[done_worlds] = -1
        
        if self.render:
            self.episode_lengths += 1
        # Override nan placeholders for alive agents
        self.buf_rews[self.dead_agent_mask] = torch.nan
        self.buf_rews[~self.dead_agent_mask] = reward[~self.dead_agent_mask]
        self.buf_dones[self.dead_agent_mask] = torch.nan
        self.buf_dones[~self.dead_agent_mask] = done[~self.dead_agent_mask].to(
            torch.float32
        )

        # Store running total reward across worlds
        self.agent_step += 1

        # Update dead agent mask: Set to True if agent is done before
        # the end of the episode
        self.dead_agent_mask = torch.logical_or(self.dead_agent_mask, done)

        # Now override the dead agent mask for the reset worlds
        if len(done_worlds) > 0:
            for world_idx in done_worlds:
                self.dead_agent_mask[
                    world_idx, :
                ] = ~self.controlled_agent_mask[world_idx, :].clone()
            self.agent_step[done_worlds] = 0

        # Construct the next observation
        next_obs = self._env.get_obs()
        self.obs_alive = next_obs[~self.dead_agent_mask]

        # RETURN NEXT_OBS, REWARD, DONE, INFO
        return (
            next_obs[self.controlled_agent_mask]
            .reshape(self.num_envs, self.obs_dim)
            .clone(),
            self.buf_rews[self.controlled_agent_mask]
            .reshape(self.num_envs)
            .clone(),
            self.buf_dones[self.controlled_agent_mask]
            .reshape(self.num_envs)
            .clone(),
            info[self.controlled_agent_mask]
            .reshape(self.num_envs, self.info_dim)
            .clone(),
        )

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def seed(self, seed=None):
        """Set the random seeds for all environments."""
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(
                np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            )

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def resample_scenario_batch(self):
        """Swap out the dataset."""

        self._env.swap_data_batch()

        # Update controlled agent mask
        self.controlled_agent_mask = self._env.cont_agent_mask.clone()
        self.max_agent_count = self._env.max_agent_count
        self.num_valid_controlled_agents_across_worlds = (
            self._env.num_valid_controlled_agents_across_worlds
        )
        self.num_envs = self.controlled_agent_mask.sum().item()

    def _update_info_dict(self, info, indices) -> None:
        """Update the info logger."""

        # Select info for controlled agents
        controlled_agent_info = info[indices][
            self.controlled_agent_mask[indices]
        ]

        self.info_dict["off_road"] = controlled_agent_info[:, 0].sum().item()
        self.info_dict["veh_collisions"] = (
            controlled_agent_info[:, 1].sum().item()
        )
        self.info_dict["non_veh_collision"] = (
            controlled_agent_info[:, 2].sum().item()
        )
        self.info_dict["goal_achieved"] = (
            controlled_agent_info[:, 3].sum().item()
        )
        self.info_dict["num_controlled_agents"] = self.controlled_agent_mask[
            indices
        ].sum()

        # Log the agents that are done but did not receive any reward
        self.info_dict["truncated"] = (
            (
                (self.agent_step[indices] == self.config.episode_len - 1)
                * ~self.dead_agent_mask[indices]
            )
            .sum()
            .item()
        )

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(
        self, method_name, *method_args, indices=None, **method_kwargs
    ):
        if "method" == "render":
            return self._env.render()
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()

    def get_images(self, policy=None) -> Sequence[Optional[np.ndarray]]:
        frames = [self._env.render()]
        return frames
    
    def render_env(self):
        """Render the environment based on conditions.
        - If the episode has just started, start a new rendering.
        - If the episode is in progress, continue rendering.
        - If the episode has ended, log the video to WandB.
        - Only render env once per rollout
        """
        for render_env_idx in range(self.render_k_scenarios):
            # Start a new rendering if the episode has just started
            if (self.iters - 1)  % self.render_interval == 0:
                if (
                    self.episode_lengths[render_env_idx] == 0
                    and not self.was_rendered_in_rollout[render_env_idx]
                ):
                    self.rendering_in_progress[render_env_idx] = True

        envs_to_render = list(
            np.where(np.array(list(self.rendering_in_progress.values())))[
                0
            ]
        )
        time_steps = list(self.episode_lengths[envs_to_render])
        
        if len(envs_to_render) > 0:
            sim_state_figures = self._env.vis.plot_simulator_state(
                env_indices=envs_to_render,
                time_steps=time_steps,
                zoom_radius=self.zoom_radius,
            )
            
            for idx, render_env_idx in enumerate(envs_to_render):
                self.frames[render_env_idx].append(
                    img_from_fig(sim_state_figures[idx])
                    )
    
    def clear_render_storage(self):
        """Clear rendering storage."""
        for env_idx in range(self.render_k_scenarios):
            self.frames[env_idx] = []
            self.rendering_in_progress[env_idx] = False
            self.was_rendered_in_rollout[env_idx] = False

    def log_video_to_wandb(self, render_env_idx, done_worlds):
        """Log arrays as videos to wandb."""
        if (
            render_env_idx in done_worlds
            and len(self.frames[render_env_idx]) > 0
        ):
            frames_array = np.array(self.frames[render_env_idx])
            self.wandb_obj.log(
                {
                    f"vis/state/env_{render_env_idx}": wandb.Video(
                        np.moveaxis(frames_array, -1, 1),
                        fps=self.render_fps,
                        format=self.render_format,
                    )
                }
            )
                
            # Reset rendering storage
            self.frames[render_env_idx] = []
            self.rendering_in_progress[render_env_idx] = False
            self.was_rendered_in_rollout[render_env_idx] = True