import os
import numpy as np
from pathlib import Path
import torch
import wandb
import gymnasium
from collections import Counter
from gpudrive.env.config import EnvConfig

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.observation import (
    LocalEgoState,
)

from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

from pufferlib.environment import PufferEnv
from gpudrive import GPU_DRIVE_DATA_DIR


def env_creator(name="gpudrive", **kwargs):
    return lambda: PufferGPUDrive(**kwargs)


class PufferGPUDrive(PufferEnv):
    """GPUDrive wrapper for PufferEnv."""

    def __init__(
        self,
        data_loader=None,
        data_dir=GPU_DRIVE_DATA_DIR,
        loader_batch_size=128,
        loader_dataset_size=3,
        loader_sample_with_replacement=True,
        loader_shuffle=False,
        device=None,
        num_worlds=128,
        max_controlled_agents=64,
        dynamics_model="classic",
        action_space_steer_disc=13,
        action_space_accel_disc=7,
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        lidar_obs=False,
        reward_type="weighted_combination",
        collision_behavior="ignore",
        collision_weight=-0.5,
        off_road_weight=-0.5,
        goal_achieved_weight=1,
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        remove_non_vehicles=True,
        obs_radius=50.0,
        render=False,
        render_interval=50,
        render_k_scenarios=3,
        render_simulator_state=True,
        render_agent_obs=False,
        render_format="mp4",
        render_fps=15,
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

        self.render = render
        self.render_interval = render_interval
        self.render_k_scenarios = render_k_scenarios
        self.render_simulator_state = render_simulator_state
        self.render_agent_obs = render_agent_obs
        self.render_format = render_format
        self.render_fps = render_fps

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
            norm_obs=norm_obs,
            dynamics_model=dynamics_model,
            collision_behavior=collision_behavior,
            dist_to_goal_threshold=dist_to_goal_threshold,
            polyline_reduction_threshold=polyline_reduction_threshold,
            remove_non_vehicles=remove_non_vehicles,
            lidar_obs=lidar_obs,
            disable_classic_obs=True if lidar_obs else False,
            obs_radius=obs_radius,
            steer_actions=torch.round(
                torch.linspace(-torch.pi, torch.pi, action_space_steer_disc), decimals=3
            ),
            accel_actions=torch.round(
                torch.linspace(-4.0, 4.0, action_space_accel_disc), decimals=3
            )
        )
        self.env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=data_loader,
            max_cont_agents=max_controlled_agents,
            device=device,
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.single_action_space = self.env.action_space
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.float32
        )

        # self.single_observation_space = self.env.single_observation_space
        # self.observation_space = self.env.observation_space
        self.controlled_agent_mask = self.env.cont_agent_mask.clone()

        # Number of controlled agents across all worlds
        self.num_agents = self.controlled_agent_mask.sum().item()

        # Reset the environment and get the initial observations
        self.observations = self.env.reset()[self.controlled_agent_mask]

        # This assigns a bunch of buffers to self.
        # You can't use them because you want torch, not numpy
        # So I am careful to assign these afterwards
        super().__init__()

        self.masks = np.ones(self.num_agents, dtype=bool)
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

        self.global_step = 0
        self.iters = 0

        # Data logging storage
        self.file_to_index = {
            file: idx for idx, file in enumerate(self.env.data_loader.dataset)
        }
        self.cumulative_unique_files = set()

    def _obs_and_mask(self, obs):
        # self.buf.masks[:] = self.env.cont_agent_mask.numpy().ravel() * self.live_agent_mask
        # return np.asarray(obs).reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)
        # return obs.numpy().reshape(NUM_WORLDS*MAX_NUM_OBJECTS, self.obs_size)[:, :6]
        return obs.view(self.total_agents, self.obs_size)

    def close(self):
        """There is no point in closing the env because
        Madrona doesn't close correctly anyways. You will want
        to cache this copy for later use. Cuda errors if you don't"""
        self.env.close()

    def reset(self, seed=None, options=None):
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

        return self.observations, []

    def step(self, action):
        """
        Step the environment with the given actions. Note that we reset worlds
        asynchronously when they are done.
        Args:
            action: A numpy array of actions for the controlled agents. Shape:
                (num_worlds, max_cont_agents_per_env)
        """

        # (0) Set the action for the controlled agents
        action = torch.from_numpy(action).to(self.device)
        self.actions[self.controlled_agent_mask] = action

        # (1) Step the simulator with controlled agents actions
        self.env.step_dynamics(self.actions)

        # Render
        self.render_env() if self.render else None

        # (2) Get rewards, terminal (dones) and info
        reward = self.env.get_rewards(
            collision_weight=self.collision_weight,
            off_road_weight=self.off_road_weight,
            goal_achieved_weight=self.goal_achieved_weight,
            world_time_steps=self.episode_lengths[:, 0].long(),
        )
        # Flatten rewards; only keep rewards for controlled agents
        reward_controlled = reward[self.controlled_agent_mask]
        terminal = self.env.get_dones().bool()

        # (3) Check if any worlds are done
        done_worlds = (
            torch.where(
                (terminal.nan_to_num(0) * self.controlled_agent_mask).sum(
                    dim=1
                )
                == self.controlled_agent_mask.sum(dim=1)
            )[0]
            .cpu()
            .numpy()
        )

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

        # (4) Use previous live agent mask as mask
        self.masks = (  # Flattend mask: (num_worlds * max_cont_agents_per_env)
            self.live_agent_mask[self.controlled_agent_mask].cpu().numpy()
        )

        # (5) Set the mask to False for _agents_ that are terminated for the next step
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

        info = []

        if len(done_worlds) > 0:

            # Log episode videos
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

            agent_episode_returns = self.agent_episode_returns[done_worlds, :][
                controlled_mask
            ]

            ego_state = LocalEgoState.from_tensor(
                self_obs_tensor=self.env.sim.self_observation_tensor(),
                backend="torch",
                device=self.device,
            )
            agent_speeds = (
                ego_state.speed[done_worlds][controlled_mask].cpu().numpy()
            )

            num_truncated = (
                truncated[done_worlds, :][controlled_mask].sum().item()
            )

            if num_finished_agents > 0:
                # fmt: off
                info.append(
                    {
                        "mean_episode_reward_per_agent": agent_episode_returns.mean().item(),
                        "perc_goal_achieved": goal_achieved_rate.item(),
                        "perc_off_road": off_road_rate.item(),
                        "perc_veh_collisions": collision_rate.item(),
                        "total_controlled_agents": self.num_agents,
                        "control_density": self.num_agents / self.controlled_agent_mask.numel(),
                        "mean_agent_speed": agent_speeds.mean().item(),
                        "episode_length": self.episode_lengths[done_worlds, :].mean().item(),
                        "num_truncated": num_truncated,
                        "perc_truncated": num_truncated / num_finished_agents,
                        "num_completed_episodes": len(done_worlds),
                    }
                )
                # fmt: on

            # Get obs for the last terminal step (before reset)
            self.last_obs = self.env.get_obs()[self.controlled_agent_mask]

            # Asynchronously reset the done worlds and empty storage
            for idx in done_worlds:
                self.env.sim.reset([idx])
                self.episode_returns[idx] = 0
                self.agent_episode_returns[idx, :] = 0
                self.episode_lengths[idx, :] = 0
                # Reset the live agent mask so that the next alive mask will mark
                # all agents as alive for the next step
                self.live_agent_mask[idx] = self.controlled_agent_mask[idx]
                self.offroad_in_episode[idx, :] = 0
                self.collided_in_episode[idx, :] = 0

        # (6) Get the next observations. Note that we do this after resetting
        # the worlds so that we always return a fresh observation
        next_obs = self.env.get_obs()[self.controlled_agent_mask]

        self.observations = next_obs
        self.rewards = reward_controlled
        self.terminals = terminal
        self.truncations = truncated[self.controlled_agent_mask]

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render_env(self):
        """Render the environment based on conditions.
        - If the episode has just started, start a new rendering.
        - If the episode is in progress, continue rendering.
        - If the episode has ended, log the video to WandB.
        - Only render env once per rollout.
        """
        for render_env_idx in range(self.render_k_scenarios):
            # Start a new rendering if the episode has just started
            if self.iters % self.render_interval == 0:
                if (
                    self.episode_lengths[render_env_idx, :][0] == 0
                    and not self.was_rendered_in_rollout[render_env_idx]
                ):
                    self.rendering_in_progress[render_env_idx] = True

        # Continue rendering if in progress
        if self.render_simulator_state:
            envs_to_render = list(
                np.where(np.array(list(self.rendering_in_progress.values())))[
                    0
                ]
            )
            time_steps = list(self.episode_lengths[envs_to_render, 0])

            sim_state_figures = self.env.vis.plot_simulator_state(
                env_indices=envs_to_render,
                time_steps=time_steps,
                zoom_radius=100,
            )
            for idx, render_env_idx in enumerate(envs_to_render):
                self.frames[render_env_idx].append(
                    img_from_fig(sim_state_figures[idx])
                )

    def render_agent_observations(self, env_idx):
        """Render a single observation."""
        agent_ids = torch.where(self.controlled_agent_mask[env_idx, :])[
            0
        ].cpu()

        img_arrays = []

        for agent_id in agent_ids:

            observation_fig, _ = self.env.vis.plot_agent_observation(
                env_idx=env_idx,
                agent_idx=agent_id.item(),
            )

            img_arrays.append(img_from_fig(observation_fig))

        return np.array(img_arrays)

    def resample_scenario_batch(self):
        """Sample and set new batch of WOMD scenarios."""

        # Swap the data batch
        self.env.swap_data_batch()

        # Update controlled agent mask and other masks
        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.num_agents = self.controlled_agent_mask.sum().item()
        self.masks = np.ones(self.num_agents, dtype=bool)
        self.agent_ids = np.arange(self.num_agents)

        self.reset()  # Reset storage
        # Get info from new worlds
        self.observations = self.env.reset()[self.controlled_agent_mask]

        self.log_data_coverage()

    def clear_render_storage(self):
        """Clear rendering storage."""
        for env_idx in range(self.render_k_scenarios):
            self.frames[env_idx] = []
            self.rendering_in_progress[env_idx] = False
            self.was_rendered_in_rollout[env_idx] = False

    def log_video_to_wandb(self, render_env_idx, done_worlds):
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
            # Reset rendering storage
            self.frames[render_env_idx] = []
            self.rendering_in_progress[render_env_idx] = False
            self.was_rendered_in_rollout[render_env_idx] = True

    def log_data_coverage(self):
        """Data coverage statistics."""

        scenario_counts = list(Counter(self.env.data_batch).values())
        scenario_unique = len(set(self.env.data_batch))

        batch_idx = {self.file_to_index[file] for file in self.env.data_batch}

        # Check how many new files are in the batch
        new_idx = batch_idx - self.cumulative_unique_files

        # Update the cumulative set
        self.cumulative_unique_files.update(new_idx)

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
                },
                step=self.global_step,
            )
