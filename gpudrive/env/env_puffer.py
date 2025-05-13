import os
import numpy as np
from pathlib import Path
import torch
import wandb
import gymnasium
from collections import Counter
from gpudrive.env.config import EnvConfig, RenderConfig

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
)

from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

from pufferlib.environment import PufferEnv
from gpudrive import GPU_DRIVE_DATA_DIR

"""Metrics from
    https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/trajectory_features.py
    implemented using numpy instead of tensorflow.
"""


def compute_displacement_error_np(pred_x, pred_y, ref_x, ref_y):
    """NumPy version of compute_displacement_error function.

    Computes displacement error (in x,y) w.r.t. a reference trajectory.

    Args:
        pred_x: The x-component of the predicted trajectories.
        pred_y: The y-component of the predicted trajectories.
        ref_x: The x-component of the reference trajectories.
        ref_y: The y-component of the reference trajectories.

    Returns:
        A float array with the same shape as all the arguments, containing
        the 2D distance between the predicted trajectories and the reference
        trajectories.
    """
    # Stack coordinates along a new axis
    pred_xy = np.stack([pred_x, pred_y], axis=-1)
    ref_xy = np.stack([ref_x, ref_y], axis=-1)

    # Calculate Euclidean distance
    return np.linalg.norm(pred_xy - ref_xy, ord=2, axis=-1)


def central_diff_np(t, pad_value):
    """NumPy version of central_diff function.

    Computes central difference along the last axis.
    """
    # Check if input is at least length 3 (needed for central diff)
    if t.shape[-1] < 3:
        result = np.full_like(t, pad_value)
        return result

    # Create proper padding tensors
    pad_shape = t.shape[:-1] + (1,)
    pad_array = np.full(pad_shape, pad_value)

    # Compute central difference for middle elements
    diff_t = (t[..., 2:] - t[..., :-2]) / 2

    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)


def central_logical_and_np(t, pad_value):
    """NumPy version of central_logical_and function."""
    # Check if input is at least length 3
    if t.shape[-1] < 3:
        result = np.full_like(t, pad_value)
        return result

    pad_shape = t.shape[:-1] + (1,)
    pad_array = np.full(pad_shape, pad_value)
    diff_t = np.logical_and(t[..., :-2], t[..., 2:])
    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)


def wrap_angle_np(angle):
    """Wraps angles in the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_kinematic_features_np(x, y, heading, seconds_per_step):
    """NumPy version of compute_kinematic_features function, for x,y only."""
    # Compute positions and displacement
    batch_shape = x.shape[:-1]
    time_steps = x.shape[-1]

    # For very short trajectories, just return placeholders
    if time_steps < 3:
        placeholder = np.full(x.shape, np.nan)
        return placeholder, placeholder, placeholder, placeholder

    # Linear speed - compute displacement first then magnitude
    dx = central_diff_np(x, pad_value=np.nan)
    dy = central_diff_np(y, pad_value=np.nan)

    # Calculate displacement norm directly (avoid reshaping issues)
    displacements = np.sqrt(dx**2 + dy**2)
    linear_speed = displacements / seconds_per_step

    # Linear acceleration (scalar)
    linear_accel = (
        central_diff_np(linear_speed, pad_value=np.nan) / seconds_per_step
    )

    # Angular speed and acceleration
    dh_step = wrap_angle_np(central_diff_np(heading, pad_value=np.nan))
    angular_speed = dh_step / seconds_per_step
    angular_accel = (
        central_diff_np(angular_speed, pad_value=np.nan) / seconds_per_step
    )

    return linear_speed, linear_accel, angular_speed, angular_accel


def compute_kinematic_validity_np(valid):
    """NumPy version of compute_kinematic_validity function."""
    speed_validity = central_logical_and_np(valid, pad_value=False)
    acceleration_validity = central_logical_and_np(
        speed_validity, pad_value=False
    )
    return speed_validity, acceleration_validity


def get_state(env, num_worlds):
    ego_state = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )
    glob_ego_pos_x = ego_state.pos_x
    glob_ego_pos_y = ego_state.pos_y
    return (
        glob_ego_pos_x[:num_worlds, :],
        glob_ego_pos_y[:num_worlds, :],
        ego_state.pos_z[:num_worlds, :],
        ego_state.rotation_angle[:num_worlds, :],
        ego_state.id[:num_worlds, :],
    )


def env_creator(name="gpudrive", **kwargs):
    return lambda: PufferGPUDrive(**kwargs)


class PufferGPUDrive(PufferEnv):
    """PufferEnv wrapper for GPUDrive."""

    def __init__(
        self,
        data_loader=None,
        data_dir=GPU_DRIVE_DATA_DIR,
        loader_batch_size=128,
        loader_dataset_size=3,
        loader_sample_with_replacement=True,
        loader_shuffle=False,
        device=None,
        num_worlds=64,
        max_controlled_agents=64,
        dynamics_model="classic",
        action_space_steer_disc=13,
        action_space_accel_disc=7,
        max_steer_angle=1.57,
        max_accel_value=4.0,
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        lidar_obs=False,
        bev_obs=False,
        add_previous_action=False,
        guidance=True,
        add_reference_pos_xy=True,
        add_reference_speed=True,
        add_reference_heading=False,
        smoothen_trajectory=False,
        reward_type="weighted_combination",
        guidance_speed_weight=0.0,
        guidance_heading_weight=0.0,
        smoothness_weight=0.0,
        condition_mode="random",
        collision_behavior="ignore",
        goal_behavior="remove",
        init_mode="all_non_trivial",
        collision_weight=-0.5,
        off_road_weight=-0.5,
        goal_achieved_weight=1,
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        guidance_dropout_prob=0.0,
        remove_non_vehicles=False,
        obs_radius=50.0,
        track_realism_metrics=False,
        track_n_worlds=2,
        render=False,
        render_3d=True,
        render_interval=50,
        render_every_t=5,
        render_k_scenarios=3,
        render_agent_idx=[0, 6],
        render_agent_obs=False,
        render_format="mp4",
        render_fps=15,
        zoom_radius=50,
        plot_guidance_pos_xy=False,
        buf=None,
        **kwargs,
    ):
        assert buf is None, "GPUDrive set up only for --vec native"

        if data_loader is None:
            data_loader = SceneDataLoader(
                root=data_dir,
                batch_size=loader_batch_size,
                dataset_size=loader_dataset_size,
                sample_with_replacement=loader_sample_with_replacement,
                shuffle=loader_shuffle,
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.num_worlds = num_worlds
        self.max_cont_agents_per_env = max_controlled_agents
        self.collision_weight = collision_weight
        self.off_road_weight = off_road_weight
        self.goal_achieved_weight = goal_achieved_weight
        self.init_mode = init_mode
        self.reward_type = reward_type

        # Expert guidance
        self.guidance = guidance
        self.add_reference_pos_xy = add_reference_pos_xy
        self.add_reference_speed = add_reference_speed
        self.add_reference_heading = add_reference_heading
        self.guidance_dropout_prob = guidance_dropout_prob

        self.render = render
        self.render_interval = render_interval
        self.render_every_t = render_every_t
        self.render_k_scenarios = render_k_scenarios
        self.render_agent_idx = render_agent_idx
        self.render_agent_obs = render_agent_obs
        self.render_format = render_format
        self.render_fps = render_fps
        self.zoom_radius = zoom_radius
        self.plot_guidance_pos_xy = plot_guidance_pos_xy
        self.track_realism_metrics = track_realism_metrics
        self.track_n_worlds = track_n_worlds

        # Total number of agents across envs, including padding
        self.total_agents = self.max_cont_agents_per_env * self.num_worlds

        # Set working directory to the base directory 'gpudrive'
        working_dir = os.path.join(Path.cwd(), "../gpudrive")
        os.chdir(working_dir)

        # Make env
        env_config = EnvConfig(
            ego_state=ego_state,
            road_map_obs=road_map_obs,
            partner_obs=partner_obs,
            reward_type=reward_type,
            guidance_speed_weight=guidance_speed_weight,
            guidance_heading_weight=guidance_heading_weight,
            smoothness_weight=smoothness_weight,
            condition_mode=condition_mode,
            norm_obs=norm_obs,
            bev_obs=bev_obs,
            add_previous_action=add_previous_action,
            guidance=guidance,
            guidance_dropout_prob=guidance_dropout_prob,
            add_reference_pos_xy=add_reference_pos_xy,
            add_reference_speed=add_reference_speed,
            add_reference_heading=add_reference_heading,
            smoothen_trajectory=smoothen_trajectory,
            dynamics_model=dynamics_model,
            collision_behavior=collision_behavior,
            goal_behavior=goal_behavior,
            init_mode=init_mode,
            dist_to_goal_threshold=dist_to_goal_threshold,
            polyline_reduction_threshold=polyline_reduction_threshold,
            remove_non_vehicles=remove_non_vehicles,
            lidar_obs=lidar_obs,
            disable_classic_obs=True if lidar_obs else False,
            obs_radius=obs_radius,
            max_steer_angle=max_steer_angle,
            max_accel_value=max_accel_value,
            action_space_steer_disc=action_space_steer_disc,
            action_space_accel_disc=action_space_accel_disc,
            # Override action space
            steer_actions = torch.round(torch.linspace(
                    -max_steer_angle, max_steer_angle, action_space_steer_disc
                ),
                decimals=3,
                ),
            accel_actions = torch.round(
                torch.linspace(
                    -max_accel_value, max_accel_value, action_space_accel_disc
                ),
                decimals=3,
            )
        )
        render_config = RenderConfig(
            render_3d=render_3d,
        )

        self.env = GPUDriveTorchEnv(
            config=env_config,
            render_config=render_config,
            data_loader=data_loader,
            max_cont_agents=max_controlled_agents,
            device=device,
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.single_action_space = self.env.action_space
        self.single_observation_space = self.env.single_observation_space

        self.controlled_agent_mask = self.env.cont_agent_mask.clone()

        # Number of controlled agents across all worlds
        self.num_agents = self.controlled_agent_mask.sum().item()

        # This assigns a bunch of buffers to self.
        # You can't use them because you want torch, not numpy
        # So I am careful to assign these afterwards
        super().__init__()

        # Reset the environment and get the initial observations
        self.observations = self.env.reset(mask=self.controlled_agent_mask)

        self.masks = torch.ones(self.num_agents, dtype=bool)
        self.actions = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env), dtype=torch.int64
        ).to(self.device)

        # Setup rendering storage
        self.rendering_in_progress = {
            env_idx: False for env_idx in range(render_k_scenarios)
        }
        self.was_rendered_in_rollout = {
            env_idx: True for env_idx in range(render_k_scenarios)
        }
        self.frames = {env_idx: [] for env_idx in range(render_k_scenarios)}
        self.agent_frames = {agent_idx: [] for agent_idx in render_agent_idx}

        self.global_step = 0
        self.iters = 0

        # Data logging storage
        self.file_to_index = {
            file: idx for idx, file in enumerate(self.env.data_loader.dataset)
        }
        self.cumulative_unique_files = set()

    def close(self):
        """There is no point in closing the env because
        Madrona doesn't close correctly anyways. You will want
        to cache this copy for later use. Cuda errors if you don't"""
        self.env.close()

    def reset(self, seed=None):
        self.rewards = torch.zeros(self.num_agents, dtype=torch.float32).to(
            self.device
        )
        self.terminals = torch.zeros(self.num_agents, dtype=torch.bool).to(
            self.device
        )
        self.truncations = torch.zeros(self.num_agents, dtype=torch.bool).to(
            self.device
        )
        self.episode_returns = torch.zeros(
            self.num_agents, dtype=torch.float32
        ).to(self.device)
        self.human_like_rewards = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32,
        ).to(self.device)
        self.internal_rewards = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32,
        ).to(self.device)
        self.agent_episode_returns = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32,
        ).to(self.device)
        self.episode_lengths = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32,
        ).to(self.device)
        self.live_agent_mask = torch.ones(
            (self.num_worlds, self.max_cont_agents_per_env), dtype=bool
        ).to(self.device)
        self.collided_in_episode = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32,
        ).to(self.device)
        self.offroad_in_episode = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32,
        ).to(self.device)

        # Storage for computing realism metrics
        if self.track_realism_metrics:
            self.worlds_to_track = self.track_n_worlds
            self.pos_xyz = torch.zeros(
                (self.worlds_to_track, self.max_cont_agents_per_env, 91, 3),
                dtype=torch.float32,
            ).to(self.device)

            self.headings = torch.zeros(
                (self.worlds_to_track, self.max_cont_agents_per_env, 91),
                dtype=torch.float32,
            ).to(self.device)

        return self.observations, []

    def step(self, action):
        """
        Step the environment with the given actions. Note that we reset worlds
        asynchronously when they are done.
        Args:
            action: A numpy array of actions for the controlled agents. Shape:
                (num_worlds, max_cont_agents_per_env)
        """

        # Set the action for the controlled agents
        self.actions[self.controlled_agent_mask] = action

        # Step the simulator with controlled agents actions
        self.env.step_dynamics(self.actions)

        # Get rewards, terminal (dones) and info
        reward = self.env.get_rewards(
            collision_weight=self.collision_weight,
            off_road_weight=self.off_road_weight,
            goal_achieved_weight=self.goal_achieved_weight,
        )

        # Flatten rewards; only keep rewards for controlled agents
        reward_controlled = reward[self.controlled_agent_mask]

        # Store human-like and internal rewards separately
        if self.reward_type == "guided_autonomy":
            self.human_like_rewards[
                self.live_agent_mask
            ] += self.env.guidance_reward[self.live_agent_mask]
            self.internal_rewards[
                self.live_agent_mask
            ] += self.env.base_rewards[self.live_agent_mask]

        terminal = self.env.get_dones()

        self.render_env() if self.render else None

        # Check if any worlds are done (terminal or truncated)
        controlled_per_world = self.controlled_agent_mask.sum(dim=1)
        done_worlds = torch.where(
            (terminal * self.controlled_agent_mask).sum(dim=1)
            == controlled_per_world
        )[0]
        done_worlds_cpu = done_worlds.cpu().numpy()

        # Add rewards for living agents
        self.agent_episode_returns[self.live_agent_mask] += reward[
            self.live_agent_mask
        ]
        self.episode_returns += reward_controlled
        self.episode_lengths += 1

        # Log off road and collision events
        info = self.env.get_infos()
        self.offroad_in_episode += info.off_road
        self.collided_in_episode += info.collided

        if self.track_realism_metrics:  # Log global states
            batch_indices = torch.arange(
                self.worlds_to_track, device=self.device
            )
            pos_x, pos_y, pos_z, heading, ids = get_state(
                self.env, num_worlds=self.worlds_to_track
            )
            episode_time_steps = (
                self.episode_lengths[: self.worlds_to_track, 0].long() - 1
            )
            self.pos_xyz[
                batch_indices, :, episode_time_steps, :
            ] = torch.stack([pos_x, pos_y, pos_z], dim=-1)
            self.headings[batch_indices, :, episode_time_steps] = heading

        # Mask used for buffer
        self.masks = self.live_agent_mask[self.controlled_agent_mask]

        # Set the mask to False for _agents_ that are terminated for the next step
        # Shape: (num_worlds, max_cont_agents_per_env)
        self.live_agent_mask[terminal] = 0

        # Truncated is defined as not crashed nor goal achieved
        truncated = torch.logical_and(
            ~self.offroad_in_episode.bool(),
            torch.logical_and(
                ~self.collided_in_episode.bool(),
                ~self.env.get_infos().goal_achieved.bool(),
            ),
        )

        # Flatten
        terminal = terminal[self.controlled_agent_mask]

        self.info_lst = []
        if len(done_worlds) > 0:

            if self.render:
                for render_env_idx in range(self.render_k_scenarios):
                    self.log_video_to_wandb(render_env_idx, done_worlds)

            # Log episode statistics
            controlled_mask = self.controlled_agent_mask[
                done_worlds, :
            ].clone()

            num_finished_agents = controlled_mask.sum().item()

            # Collision rates are summed across all agents in the episode
            off_road_rate = (
                torch.where(
                    self.offroad_in_episode[done_worlds, :][controlled_mask]
                    > 0,
                    1,
                    0,
                ).sum()
                / num_finished_agents
            )
            collision_rate = (
                torch.where(
                    self.collided_in_episode[done_worlds, :][controlled_mask]
                    > 0,
                    1,
                    0,
                ).sum()
                / num_finished_agents
            )
            goal_achieved_rate = (
                self.env.get_infos()
                .goal_achieved[done_worlds, :][controlled_mask]
                .sum()
                / num_finished_agents
            )

            # Calculate human-likeness metrics for completed episodes
            human_like_values = self.human_like_rewards[done_worlds, :][
                controlled_mask
            ]
            internal_reward_values = self.internal_rewards[done_worlds, :][
                controlled_mask
            ]

            agent_episode_returns = self.agent_episode_returns[done_worlds, :][
                controlled_mask
            ]

            num_truncated = (
                truncated[done_worlds, :][controlled_mask].sum().item()
            )

            if num_finished_agents > 0:
                # fmt: off
                self.info_lst.append(
                    {
                        "metrics/mean_episode_reward_per_agent": agent_episode_returns.mean().item(),
                        "metrics/mean_guidance_reward": human_like_values.mean().item(),
                        "metrics/mean_internal_reward": internal_reward_values.mean().item(),
                        "metrics/perc_goal_achieved": goal_achieved_rate.item(),
                        "metrics/perc_off_road": off_road_rate.item(),
                        "metrics/perc_collisions": collision_rate.item(),
                        "metrics/perc_truncated": num_truncated / num_finished_agents,
                        "train/num_completed_episodes": len(done_worlds),
                        "train/total_controlled_agents": self.num_agents,
                        "train/control_density": self.num_agents / self.controlled_agent_mask.numel(),
                        "train/episode_length": self.episode_lengths[done_worlds, :].mean().item(),
                    }
                )
                # fmt: on

            # Get obs for the last terminal step (before reset)
            self.last_obs = self.env.get_obs(self.controlled_agent_mask)

            # Asynchronously reset the done worlds and empty storage
            self.env.reset(
                env_idx_list=done_worlds_cpu, mask=self.controlled_agent_mask
            )
            self.episode_returns[done_worlds] = 0
            self.agent_episode_returns[done_worlds, :] = 0
            self.episode_lengths[done_worlds, :] = 0
            # Reset the live agent mask so that the next alive mask will mark
            # all agents as alive for the next step
            self.live_agent_mask[done_worlds] = self.controlled_agent_mask[
                done_worlds
            ]
            self.offroad_in_episode[done_worlds, :] = 0
            self.collided_in_episode[done_worlds, :] = 0
            self.human_like_rewards[done_worlds, :] = 0
            self.internal_rewards[done_worlds, :] = 0

            if self.track_realism_metrics:
                tracked_done_worlds = done_worlds.tolist() and list(
                    range(self.worlds_to_track)
                )
                if len(tracked_done_worlds):
                    self.compute_realism_metrics(tracked_done_worlds)
                    self.pos_xyz[tracked_done_worlds, :] = 0
                    self.headings[tracked_done_worlds, :] = 0

        # Get the next observations. Note that we do this after resetting
        # the worlds so that we always return a fresh observation
        next_obs = self.env.get_obs(self.controlled_agent_mask)

        self.observations = next_obs
        self.rewards = reward_controlled
        self.terminals = terminal
        self.truncations = truncated[self.controlled_agent_mask]

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            self.info_lst,
        )

    def render_env(self):
        """Render the environment based on conditions.
        - If the episode has just started, start a new rendering.
        - If the episode is in progress, continue rendering.
        - If the episode has ended, log the video to WandB.
        - Only render env once per rollout
        """
        for render_env_idx in range(self.render_k_scenarios):
            # Start a new rendering if the episode has just started
            if (self.iters - 1) % self.render_interval == 0:
                if (
                    self.episode_lengths[render_env_idx, :][0] == 0
                    and not self.was_rendered_in_rollout[render_env_idx]
                ):
                    self.rendering_in_progress[render_env_idx] = True

        envs_to_render = list(
            np.where(np.array(list(self.rendering_in_progress.values())))[0]
        )
        time_steps = list(self.episode_lengths[envs_to_render, 0])

        render_at_time = bool(
            self.env.step_in_world[0, 0] % self.render_every_t == 0
        )

        if len(envs_to_render) > 0 and render_at_time:
            sim_state_figures = self.env.vis.plot_simulator_state(
                env_indices=envs_to_render,
                time_steps=time_steps,
                zoom_radius=self.zoom_radius,
                plot_guidance_pos_xy=self.plot_guidance_pos_xy,
            )

            for agent_idx in self.render_agent_idx:
                agent_obs = self.env.vis.plot_agent_observation(
                    env_idx=0,
                    agent_idx=0,
                    figsize=(10, 10),
                    trajectory=self.env.reference_path[agent_idx, :, :].to(
                        "cpu"
                    ),
                    step_reward=self.env.guidance_reward[0, agent_idx].item(),
                    route_progress=self.env.route_progress[agent_idx].item(),
                )
                self.agent_frames[agent_idx].append(img_from_fig(agent_obs))

            for idx, render_env_idx in enumerate(envs_to_render):
                self.frames[render_env_idx].append(
                    img_from_fig(sim_state_figures[idx])
                )

    def resample_scenario_batch(self):
        """Sample and set new batch of WOMD scenarios."""

        # Swap the data batch
        self.env.swap_data_batch()

        # Update controlled agent mask and other masks
        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.num_agents = self.controlled_agent_mask.sum().item()
        self.masks = torch.ones(self.num_agents, dtype=bool)
        self.agent_ids = np.arange(self.num_agents)

        self.reset()  # Reset storage
        # Get info from new worlds
        self.observations = self.env.reset(mask=self.controlled_agent_mask)

        self.log_data_coverage()

    def clear_render_storage(self, env_list_to_clear=None):
        """Clear rendering storage."""
        for env_idx in env_list_to_clear:
            self.frames[env_idx] = []
            self.rendering_in_progress[env_idx] = False
            self.was_rendered_in_rollout[env_idx] = False

        if env_list_to_clear is not None:
            for agent_idx in self.render_agent_idx:
                self.agent_frames[agent_idx] = []

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
                        caption=f"global step: {self.global_step:,}",
                    )
                }
            )
            for agent_idx in self.render_agent_idx:
                agent_frames_array = np.array(self.agent_frames[agent_idx])
                self.wandb_obj.log(
                    {
                        f"vis/agent_obs/env_0_agent_{agent_idx}": wandb.Video(
                            np.moveaxis(agent_frames_array, -1, 1),
                            fps=self.render_fps,
                            format=self.render_format,
                            caption=f"global step: {self.global_step:,}",
                        )
                    }
                )

            # Reset rendering storage
            self.frames[render_env_idx] = []
            for agent_idx in self.render_agent_idx:
                self.agent_frames[agent_idx] = []
            self.rendering_in_progress[render_env_idx] = False
            self.was_rendered_in_rollout[render_env_idx] = True

    def log_data_coverage(self):
        """Data coverage statistics."""

        scenario_counts = list(Counter(self.env.data_batch).values())
        scenario_unique = len(set(self.env.data_batch))

        batch_idx = {self.file_to_index[file] for file in self.env.data_batch}

        # Check how many new files are in the batch
        new_idx = batch_idx - self.cumulative_unique_files

        # Update the cumulative set (coverage)
        self.cumulative_unique_files.update(new_idx)

        guidance_density = (
            self.env.valid_guidance_points / self.env.reference_traj_len
        )

        if self.wandb_obj is not None:
            self.wandb_obj.log(
                {
                    "data/new_files_in_batch": len(new_idx),
                    "data/unique_scenarios_in_batch": scenario_unique,
                    "data/scenario_counts_in_batch": wandb.Histogram(
                        scenario_counts
                    ),
                    "data/coverage": (
                        len(self.cumulative_unique_files)
                        / len(set(self.file_to_index))
                    )
                    * 100,
                    "data/guidance_density_mean": guidance_density.mean().item(),
                    "data/guidance_density_dist": wandb.Histogram(
                        guidance_density.cpu().numpy()
                    ),
                },
                step=self.global_step,
            )

    def compute_realism_metrics(self, done_worlds):
        """Compute realism metrics.

        Args:
            done_worlds: List of indices of the worlds to track.
        """

        # [worlds, max_cont_agents]
        control_mask = (
            self.controlled_agent_mask[done_worlds].detach().cpu().numpy()
        )

        if control_mask.sum() > 0:
            # [batch, time, 1]
            valid_mask = (
                self.env.reference_trajectory.valids[done_worlds]
                .detach()
                .cpu()
                .numpy()[control_mask]
                .squeeze(-1)
            ).astype(bool)

            # Take human logs (ground-truth)
            # Shape: [worlds, max_cont_agents, time, 2] -> [batch, time, 2]
            ref_pos_xy_np = (
                self.env.reference_trajectory.pos_xy[done_worlds]
                .detach()
                .cpu()
                .numpy()[control_mask]
            )
            # Shape: [worlds, max_cont_agents, time, 1] -> [batch, time, 1]
            ref_headings_np = (
                self.env.reference_trajectory.yaw[done_worlds]
                .detach()
                .cpu()
                .numpy()[control_mask]
                .squeeze(-1)
            )

            agent_headings_np = (
                self.headings[done_worlds].detach().cpu().numpy()[control_mask]
            )
            agent_pos_xyz_np = (
                self.pos_xyz[done_worlds].detach().cpu().numpy()[control_mask]
            )

            # Extract x, y components (z is not informative)
            agent_x_np = agent_pos_xyz_np[..., 0]
            agent_y_np = agent_pos_xyz_np[..., 1]

            ref_x_np = ref_pos_xy_np[..., 0]
            ref_y_np = ref_pos_xy_np[..., 1]

            # Step duration in seconds
            seconds_per_step = 0.1  # Assuming 10Hz sampling rate

            # Compute the metrics for agent trajectories
            (
                agent_linear_speed,
                agent_linear_accel,
                agent_angular_speed,
                agent_angular_accel,
            ) = compute_kinematic_features_np(
                agent_x_np, agent_y_np, agent_headings_np, seconds_per_step
            )

            # Compute the metrics for reference trajectories
            (
                ref_linear_speed,
                ref_linear_accel,
                ref_angular_speed,
                ref_angular_accel,
            ) = compute_kinematic_features_np(
                ref_x_np, ref_y_np, ref_headings_np, seconds_per_step
            )

            # Check which time steps are valid
            speed_valid, accel_valid = compute_kinematic_validity_np(
                valid_mask
            )

            # Calculate absolute diffs between agent and reference trajectories
            linear_speed_error = np.abs(agent_linear_speed - ref_linear_speed)
            linear_accel_error = np.abs(agent_linear_accel - ref_linear_accel)
            angular_speed_error = np.abs(
                agent_angular_speed - ref_angular_speed
            )
            angular_accel_error = np.abs(
                agent_angular_accel - ref_angular_accel
            )

            # Calculate displacement error
            displacement_error = compute_displacement_error_np(
                agent_x_np, agent_y_np, ref_x_np, ref_y_np
            )

            # Apply validity masks
            masked_speed_error = np.ma.array(
                linear_speed_error, mask=~speed_valid
            )
            masked_accel_error = np.ma.array(
                linear_accel_error, mask=~accel_valid
            )
            masked_angular_speed_error = np.ma.array(
                angular_speed_error, mask=~speed_valid
            )
            masked_angular_accel_error = np.ma.array(
                angular_accel_error, mask=~accel_valid
            )

            # Compute MAEs
            mean_linear_speed_error = masked_speed_error.mean()
            mean_linear_accel_error = masked_accel_error.mean()
            mean_angular_speed_error = masked_angular_speed_error.mean()
            mean_angular_accel_error = masked_angular_accel_error.mean()

            self.info_lst.append(
                {
                    "realism/kinematic_linear_speed_mae": mean_linear_speed_error,
                    "realism/kinematic_linear_accel_mae": mean_linear_accel_error,
                    "realism/kinematic_angular_speed_mae": mean_angular_speed_error,
                    "realism/kinematic_angular_accel_mae": mean_angular_accel_error,
                    "realism/mean_displacement_error": displacement_error[
                        valid_mask
                    ].mean(),
                    "realism/kinematic_ref_linear_speed_dist": wandb.Histogram(
                        ref_linear_speed[speed_valid]
                    ),
                    "realism/kinematic_ref_linear_accel_dist": wandb.Histogram(
                        ref_linear_accel[accel_valid]
                    ),
                    "realism/kinematic_agent_linear_speed_dist": wandb.Histogram(
                        agent_linear_speed[speed_valid]
                    ),
                    "realism/kinematic_agent_linear_accel_dist": wandb.Histogram(
                        agent_linear_speed[accel_valid]
                    ),
                },
            )
