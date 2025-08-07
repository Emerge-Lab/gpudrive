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
)

from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

from pufferlib.environment import PufferEnv
from gpudrive import GPU_DRIVE_DATA_DIR


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
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        lidar_obs=False,
        bev_obs=False,
        reward_type="weighted_combination",
        collision_behavior="ignore",
        collision_weight=-0.5,
        off_road_weight=-0.5,
        goal_achieved_weight=1,
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        remove_non_vehicles=True,
        obs_radius=50.0,
        use_vbd=False,
        vbd_model_path=None,
        vbd_trajectory_weight=0.1,
        render=False,
        render_3d=True,
        render_interval=50,
        render_k_scenarios=3,
        render_agent_obs=False,
        render_format="mp4",
        render_fps=15,
        zoom_radius=50,
        minimum_frames_to_log=50,
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
        self.render_agent_obs = render_agent_obs
        self.render_format = render_format
        self.render_fps = render_fps
        self.zoom_radius = zoom_radius
        self.minimum_frames_to_log = minimum_frames_to_log

        # VBD
        self.vbd_model_path = vbd_model_path
        self.vbd_trajectory_weight = vbd_trajectory_weight
        self.use_vbd = use_vbd
        self.vbd_trajectory_weight = vbd_trajectory_weight

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
            bev_obs=bev_obs,
            dynamics_model=dynamics_model,
            collision_behavior=collision_behavior,
            dist_to_goal_threshold=dist_to_goal_threshold,
            polyline_reduction_threshold=polyline_reduction_threshold,
            remove_non_vehicles=remove_non_vehicles,
            lidar_obs=lidar_obs,
            disable_classic_obs=True if lidar_obs else False,
            obs_radius=obs_radius,
            steer_actions=torch.round(
                torch.linspace(-torch.pi, torch.pi, action_space_steer_disc),
                decimals=3,
            ),
            accel_actions=torch.round(
                torch.linspace(-4.0, 4.0, action_space_accel_disc), decimals=3
            ),
            use_vbd=use_vbd,
            vbd_model_path=vbd_model_path,
            vbd_trajectory_weight=vbd_trajectory_weight,
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
        self.observations = self.env.reset(self.controlled_agent_mask)

        self.masks = torch.ones(self.num_agents, dtype=bool)
        self.world_size = self.controlled_agent_mask.shape[1]
        # Action tensor must match simulator's expected shape: (num_worlds, max_num_agents_in_scene)
        # The simulator will only use actions for agents marked as controlled in cont_agent_mask
        self.actions = torch.zeros(
            (self.num_worlds, self.world_size), dtype=torch.int64
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
        self.agent_episode_returns = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.float32,
        ).to(self.device)
        self.episode_lengths = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.float32,
        ).to(self.device)
        self.live_agent_mask = torch.ones(
            (self.num_worlds, self.world_size), dtype=bool
        ).to(self.device)
        self.collided_in_episode = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.float32,
        ).to(self.device)
        self.offroad_in_episode = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.float32,
        ).to(self.device)

        self.initialize_tracking()

        return self.observations, []

    def initialize_tracking(self):
        self.done_or_truncated_worlds = torch.zeros(self.num_worlds, dtype=torch.int32).to(self.device)
        self.goal_achieved_mask = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.int32
        ).to(self.device)
        self.collided_mask = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.int32
        ).to(self.device)
        self.offroad_mask = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.int32
        ).to(self.device)
        self.truncated_mask = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.int32
        ).to(self.device)
        self.reward_agent = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.float32
        ).to(self.device)
        self.episode_length_agent = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.float32
        ).to(self.device)
        self.total_offroad_count = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.int32
        ).to(self.device)
        self.total_collided_count = torch.zeros(
            (self.num_worlds, self.world_size),
            dtype=torch.int32
        ).to(self.device)


    def step(self, action):
        """
        Step the environment with the given actions. Note that we reset worlds
        asynchronously when they are done.
        Args:
            action: A numpy array of actions for the controlled agents. Shape:
                (total_controlled_agents,) - will be mapped to controlled positions
                in the (num_worlds, max_num_agents_in_scene) action tensor
        """

        # Set the action for the controlled agents
        # print(f"action shape: {action.shape}")
        # print(f"self.controlled_agent_mask shape: {self.controlled_agent_mask.shape}")
        # print(f"total controlled agents: {self.controlled_agent_mask.sum().item()}")
        self.actions[self.controlled_agent_mask] = action

        # Step the simulator with controlled agents actions
        self.env.step_dynamics(self.actions)

        # Get rewards, terminal (dones) and info
        reward = self.env.get_rewards(
            collision_weight=self.collision_weight,
            off_road_weight=self.off_road_weight,
            goal_achieved_weight=self.goal_achieved_weight,
            world_time_steps=self.episode_lengths[:, 0].long(),
        )
        # Flatten rewards; only keep rewards for controlled agents
        reward_controlled = reward[self.controlled_agent_mask]
        terminal = self.env.get_dones().bool()

        self.render_env() if self.render else None

        # Check if any worlds are done (terminal or truncated)
        controlled_per_world = self.controlled_agent_mask.sum(dim=1)
        
        # Worlds where all controlled agents are terminal
        terminal_done_worlds = torch.where(
            (terminal * self.controlled_agent_mask).sum(dim=1)
            == controlled_per_world
        )[0]
        
        # Worlds where episodes have reached maximum length (truncated)
        max_episode_length = self.env.episode_len
        truncated_done_worlds = torch.where(
            self.episode_lengths[:, 0] >= max_episode_length
        )[0]
        
        # Combine both types of done worlds
        if len(terminal_done_worlds) > 0 and len(truncated_done_worlds) > 0:
            done_worlds = torch.unique(torch.cat([terminal_done_worlds, truncated_done_worlds]))
        elif len(terminal_done_worlds) > 0:
            done_worlds = terminal_done_worlds
        elif len(truncated_done_worlds) > 0:
            done_worlds = truncated_done_worlds
        else:
            done_worlds = torch.tensor([], dtype=torch.long, device=self.device)
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

        # Mask used for buffer
        self.masks = self.live_agent_mask[self.controlled_agent_mask]

        # Set the mask to False for _agents_ that are terminated for the next step
        # Shape: (num_worlds, world_size)
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

        info_lst = []

        if self.render:
            for render_env_idx in range(self.render_k_scenarios):
                self.log_video_to_wandb(render_env_idx, done_worlds)

        if(len(done_worlds) > 0):
            self.done_or_truncated_worlds[done_worlds] = 1
            done_world_mask = torch.zeros_like(self.controlled_agent_mask, dtype=torch.bool)
            done_world_mask[done_worlds, :] = True
            combined_mask = done_world_mask & self.controlled_agent_mask
            
            # Now use the combined mask for proper assignment
            self.goal_achieved_mask[combined_mask] = torch.where(
                self.env.get_infos().goal_achieved[combined_mask].to(torch.int32) > 0,
                torch.tensor(1, dtype=torch.int32),
                torch.tensor(0, dtype=torch.int32),
            )
            self.collided_mask[combined_mask] = torch.where(
                self.collided_in_episode[combined_mask].to(torch.int32) > 0,
                torch.tensor(1, dtype=torch.int32), 
                torch.tensor(0, dtype=torch.int32),
            )
            self.offroad_mask[combined_mask] = torch.where(
                self.offroad_in_episode[combined_mask].to(torch.int32) > 0,
                torch.tensor(1, dtype=torch.int32), 
                torch.tensor(0, dtype=torch.int32),
            )
            self.total_collided_count[combined_mask] = self.collided_in_episode[combined_mask].sum().to(torch.int32)
            self.total_offroad_count[combined_mask] = self.offroad_in_episode[combined_mask].sum().to(torch.int32)
            
            self.truncated_mask[combined_mask] = truncated[combined_mask].to(torch.int32)
            self.reward_agent[combined_mask] = self.agent_episode_returns[combined_mask]
            self.episode_length_agent[combined_mask] = self.episode_lengths[combined_mask]

            # reset the done_worlds
            # Get obs for the last terminal step (before reset)
            self.last_obs = self.env.get_obs(self.controlled_agent_mask)

            # Asynchronously reset the done worlds and empty storage
            self.env.reset(env_idx_list=done_worlds_cpu)
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
            
        if(self.done_or_truncated_worlds.sum().item() == self.num_worlds):
            # we have finished all synced worlds, now we can log the data
            goal_achieved_rate = self.goal_achieved_mask.sum() / self.num_agents
            off_road_rate = self.offroad_mask.sum() / self.num_agents
            collision_rate = self.collided_mask.sum() / self.num_agents
            truncated_rate = self.truncated_mask.sum() / self.num_agents
            crashed = self.collided_mask | self.offroad_mask
            crashed_rate = crashed.sum() / self.num_agents
            mean_episode_reward = self.reward_agent.sum() / self.num_agents
            
            # print(f"mean episode reward per agent: {mean_episode_reward.item()}")
            # print(f"goal_achieved_rate: {goal_achieved_rate.item()}, off_road_rate: {off_road_rate.item()}, collision_rate: {collision_rate.item()}, truncated_rate: {truncated_rate.item()}, PercentCrashedorGoalAchievedorTruncated: {goal_achieved_rate.item() + crashed_rate.item() + truncated_rate.item()}")

            info_lst.append(
                {
                    "perc_goal_achieved": goal_achieved_rate.item(),
                    "perc_crashed(collided or offroad)": crashed_rate.item(),
                    "perc_off_road": off_road_rate.item(),
                    "perc_veh_collisions": collision_rate.item(),
                    "perc_truncated": truncated_rate.item(),
                    "mean_episode_reward_per_agent": mean_episode_reward.item(),
                    "episode_length": self.episode_length_agent.mean().item(),
                    "total_offroad_count": self.total_offroad_count.sum().item(),
                    "total_collided_count": self.total_collided_count.sum().item(),
                }
            )

            # reset the tracking variables
            self.initialize_tracking()

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
            info_lst,
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

        if len(envs_to_render) > 0:
            sim_state_figures = self.env.vis.plot_simulator_state(
                env_indices=envs_to_render,
                time_steps=time_steps,
                zoom_radius=self.zoom_radius,
            )

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
        self.observations = self.env.reset(self.controlled_agent_mask)

        self.log_data_coverage()

    def clear_render_storage(self):
        """Clear rendering storage."""
        for env_idx in range(self.render_k_scenarios):
            self.frames[env_idx] = []
            self.rendering_in_progress[env_idx] = False
            self.was_rendered_in_rollout[env_idx] = False

    def log_video_to_wandb(self, render_env_idx, done_worlds):
        """Log arrays as videos to wandb."""
        # if(len(self.frames[render_env_idx]) > 0):
        #     print(f"iter: {self.iters}, render_env_idx: {render_env_idx}, frames length: {len(self.frames[render_env_idx])}, done_worlds: {done_worlds}")
        if (
            (render_env_idx in done_worlds and len(self.frames[render_env_idx]) > 0) 
            or len(self.frames[render_env_idx]) > self.minimum_frames_to_log
        ):
            frames_array = np.array(self.frames[render_env_idx])
            # print(f"frames shape: {frames_array.shape}")
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

        # Update the cumulative set (coverage)
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