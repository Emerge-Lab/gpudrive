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
from gpudrive.env import constants

def env_creator(name="gpudrive", **kwargs):
    return lambda: PufferGPUDrive(**kwargs)

class HistoryManager:
    def __init__(self, history_config, controlled_mask ,partner_obs_shape , device):
        self.k_trials = history_config.get("trials", 1)
        self.log_history_step = history_config.get("log_history_step", 1)
        self.episode_len = history_config.get("episode_len", 91)
        self.partner_obs_shape = partner_obs_shape
        controlled_indices = torch.nonzero(controlled_mask).squeeze()
        controlled_keys = [tuple(indice.tolist()) for indice in controlled_indices]

        self.history_dicts = {
            (world_idx, agent_idx): {
                f"trial_{k}": torch.full(
                    (int(self.episode_len/self.log_history_step)+1, self.partner_obs_shape), 
                    0.0, 
                    device=device
                )
                for k in range(self.k_trials)
            }
            for (world_idx, agent_idx) in controlled_keys
        }


    def _get_live_keys(self, live_mask):
        """Extract live keys once and reuse across methods"""
        live_indices = torch.nonzero(live_mask, as_tuple=False).view(-1, live_mask.dim())
        return [tuple(idx.tolist()) for idx in live_indices]

    def update_history(self, partner_obs, live_mask, env_to_step_in_trial_dict ):
        all_partner_observations = partner_obs
        live_keys = self._get_live_keys(live_mask)
        
        for world_key, agent_key in live_keys:
            current_step = env_to_step_in_trial_dict[f'world_{world_key}']['time_step']
            if current_step % self.log_history_step == 0:
                trial = env_to_step_in_trial_dict[f'world_{world_key}']['trial']
                step_index = current_step // self.log_history_step
                self.history_dicts[(world_key, agent_key)][f"trial_{trial}"][step_index] = \
                    all_partner_observations[world_key][agent_key]

    def get_obs_and_history(self, obs, live_mask):
        live_keys = self._get_live_keys(live_mask)
        
        if not live_keys:
            return torch.tensor([])
        
        observations_list = []
        
        # Pre-generate trial keys for reuse
        trial_keys = [f"trial_{k}" for k in range(self.k_trials)]
        
        for world_key, agent_key in live_keys:
            history_dict = self.history_dicts[(world_key, agent_key)]
            
            # More efficient tensor stacking
            history_tensors = [history_dict[trial_key] for trial_key in trial_keys]
            history_batch = torch.stack(history_tensors)
            flattened_history_batch = history_batch.flatten()
            
            # Combine current observation with history
            combined_obs = torch.cat((obs[world_key][agent_key], flattened_history_batch))
            observations_list.append(combined_obs)
        
        return torch.vstack(observations_list)

    def clear_history(self, world_indices):
        """
        Clear history for all agents in specified worlds by zeroing out tensors
        
        Args:
            world_indices: torch.Tensor containing world indices to clear
        """
        if not isinstance(world_indices, torch.Tensor):
            world_indices = torch.tensor(world_indices)
        
        worlds_to_clear = set(world_indices.tolist())
        
        # Only iterate through keys that match the worlds to clear
        for key in self.history_dicts.keys():
            if key[0] in worlds_to_clear:  # key[0] is world_key
                for trial_key in self.history_dicts[key].keys():
                    self.history_dicts[key][trial_key].zero_()

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
        history_config=None,
        hide_reward_conditioning = True,
        condition_mode = 'random',
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
        self.hide_reward_conditioning = hide_reward_conditioning

        # VBD
        self.vbd_model_path = vbd_model_path
        self.vbd_trajectory_weight = vbd_trajectory_weight
        self.use_vbd = use_vbd
        self.vbd_trajectory_weight = vbd_trajectory_weight

        # Total number of agents across envs, including padding
        self.total_agents = self.max_cont_agents_per_env * self.num_worlds

        # Set working directory to the base directory 'gpudrive'
        working_dir = Path.cwd()/ "gpudrive"
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
            condition_mode=condition_mode
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


        self.env_to_step_in_trial_dict = {
            f"world_{w}": {'trial': 0, 'time_step': 0 } for w in range(self.num_worlds)
        }

        # Number of controlled agents across all worlds
        self.num_agents = self.controlled_agent_mask.sum().item()

        if history_config is not None:
            partner_obs_shape = self.env._get_partner_obs().shape[-1]
            self.history_manager = HistoryManager(
                history_config=history_config,
                controlled_mask=self.controlled_agent_mask,
                partner_obs_shape=partner_obs_shape,
                device=self.device
            )
            current_obs_space = self.single_observation_space

            low = current_obs_space.low
            high = current_obs_space.high
            if history_config.closest_k_partners_in_history =='all':
                new_dim = self.env.single_observation_space.shape[0] + history_config.trials*(int(history_config.num_steps/history_config.log_history_step)+1)*partner_obs_shape
            else:
                new_dim = self.env.single_observation_space.shape[0] + history_config.trials*(int(history_config.num_steps/history_config.log_history_step)+1)*constants.PARTNER_FEAT_DIM *history_config.closest_k_partners_in_history
            new_low = np.full((new_dim,), -1.0, dtype=np.float32)
            new_high = np.full((new_dim,), 1.0, dtype=np.float32)
            
            self.single_observation_space = gymnasium.spaces.Box(low=new_low,high=new_high,shape=(new_dim,),dtype=np.float32)

        # This assigns a bunch of buffers to self.
        # You can't use them because you want torch, not numpy
        # So I am careful to assign these afterwards
        super().__init__()

        # Reset the environment and get the initial observations
        self.observations = self.env.reset()
        if hasattr(self,'history_manager'):
            self.assign_vehicles_to_actors_all_worlds()
            if self.hide_reward_conditioning: 
                ego_obs = self.observations[self.ego_agent_mask]
                reward_weights = ego_obs[:, 6:9]
                ego_obs[:, 6:9] = -99999999 ### very low number to indicate hidden reward conditioning
                self.observations[self.ego_agent_mask] = ego_obs
            self.observations = self.history_manager.get_obs_and_history(
                self.observations, self.controlled_agent_mask
            )
            



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

        # Set the action for the controlled agents
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
        done_worlds = torch.where(
            (terminal * self.controlled_agent_mask).sum(dim=1)
            == controlled_per_world
        )[0]
        done_worlds_cpu = done_worlds.cpu().numpy()

        for world,num_agents in enumerate(controlled_per_world):
            if num_agents > 0:  
                
                self.env_to_step_in_trial_dict[f"world_{world}"]['time_step'] += 1
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

        info_lst = []
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

            total_collisions = self.collided_in_episode[done_worlds, :].sum()
            total_off_road = self.offroad_in_episode[done_worlds, :].sum()

            agent_episode_returns = self.agent_episode_returns[done_worlds, :][
                controlled_mask
            ]

            num_truncated = (
                truncated[done_worlds, :][controlled_mask].sum().item()
            )

            if num_finished_agents > 0:
                # fmt: off
                info_lst.append(
                    {
                        "mean_episode_reward_per_agent": agent_episode_returns.mean().item(),
                        "perc_goal_achieved": goal_achieved_rate.item(),
                        "perc_off_road": off_road_rate.item(),
                        "perc_veh_collisions": collision_rate.item(),
                        "total_controlled_agents": self.num_agents,
                        "control_density": self.num_agents / self.controlled_agent_mask.numel(),
                        "episode_length": self.episode_lengths[done_worlds, :].mean().item(),
                        "perc_truncated": num_truncated / num_finished_agents,
                        "num_completed_episodes": len(done_worlds),
                        "total_collisions": total_collisions.item(),
                        "total_off_road": total_off_road.item(),
                    }
                )
                # fmt: on

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

            for world_idx in done_worlds_cpu:
                trial = self.env_to_step_in_trial_dict[f"world_{world_idx}"]['trial']
                if trial == self.history_manager.k_trials - 1:
                    if hasattr(self, 'history_manager'):
                        self.history_manager.clear_history([world_idx])
                        self.assign_vehilces_to_actors_done_worlds([world_idx])
                    self.env_to_step_in_trial_dict[f"world_{world_idx}"] = {
                        'trial': 0,
                        'time_step': 0
                    }
                else:
                    self.env_to_step_in_trial_dict[f"world_{world_idx}"]['trial'] += 1
                    self.env_to_step_in_trial_dict[f"world_{world_idx}"]['time_step'] = 0


        # Get the next observations. Note that we do this after resetting
        # the worlds so that we always return a fresh observation
        next_obs = self.env.get_obs()
        if hasattr(self,'history_manager'):
            if self.hide_reward_conditioning: 
                ego_obs = next_obs[self.ego_agent_mask]
                reward_weights = ego_obs[:, 6:9]
                ego_obs[:, 6:9] = -99999999 ### very low number to indicate hidden reward conditioning
                next_obs[self.ego_agent_mask] = ego_obs
            
            next_obs = self.history_manager.get_obs_and_history(
                next_obs, self.controlled_agent_mask , 
            )

            partner_obs = self.env._get_partner_obs()
            self.history_manager.update_history(
                partner_obs, self.controlled_agent_mask, self.env_to_step_in_trial_dict
            )
        else:
            next_obs = next_obs[self.controlled_agent_mask]

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

    def assign_vehicles_to_actors_all_worlds(self):
        """Tag one vehicle per scene as av agent and the rest as co-players."""

        self.controllable_agent_idx = self.controlled_agent_mask.nonzero(
            as_tuple=False
        )
            
        self.ego_agent_mask = torch.zeros_like(
            self.controlled_agent_mask, dtype=torch.bool
        )

        sampled_agent_indices = self.sample_vehicles_in_all_worlds()

        self.create_masks(sampled_agent_indices)

    def sample_vehicles_in_all_worlds(self):
    
        environments = self.controllable_agent_idx[:, 0]
        
        self.controllable_worlds, inverse = torch.unique(environments, return_inverse=True)
        full_range = torch.arange(self.controlled_agent_mask.shape[0]).to(self.device)
        self.missing_worlds = full_range[~torch.isin(full_range, self.controllable_worlds)]
        
        # Double-check with controlled agent mask
        number_controllable_agents_per_world = self.controlled_agent_mask.sum(dim=1)
        self.non_controllable_worlds = torch.where(number_controllable_agents_per_world == 0)[0]
        # Verify the two methods of identifying non-controllable worlds match
        assert torch.all(self.missing_worlds == self.non_controllable_worlds), "mismatch between missing worlds and non-controllable worlds"
        assert(len(self.non_controllable_worlds)==0)
        
        # Initialize tensor to store sampled indices for all worlds
        sampled_indices = torch.full(
            (self.num_worlds, 2),
            -1,
            device=self.device,
            dtype=self.controllable_agent_idx.dtype,
        )

        for world in self.controllable_worlds:

            mask = environments == world
            indices_in_env = self.controllable_agent_idx[mask]
            
            # Sample according to specified method
            if indices_in_env.size(0) > 0:

                random_index = torch.randint(0, indices_in_env.size(0), (1,)).item()
                sampled_indices[world] = indices_in_env[random_index]

        valid_rows = (sampled_indices[:, 0] >= 0)
        return sampled_indices[valid_rows]
    
    def create_masks(self,sampled_agent_indices):

        max_idx_0 = self.ego_agent_mask.shape[0] - 1
        max_idx_1 = self.ego_agent_mask.shape[1] - 1

        valid_indices = (
            (sampled_agent_indices[:, 0] >= 0) & 
            (sampled_agent_indices[:, 0] <= max_idx_0) &
            (sampled_agent_indices[:, 1] >= 0) & 
            (sampled_agent_indices[:, 1] <= max_idx_1)
        )

        valid_sampled_indices = sampled_agent_indices[valid_indices]
        
        # Instead of setting all masks to False, only update masks for worlds in valid_sampled_indices
        worlds_to_update = valid_sampled_indices[:, 0].unique()
        
        # Only set masks to False for the worlds we're updating
        for world_idx in worlds_to_update:
            self.ego_agent_mask[world_idx] = False
            
        # Set specific indices to True
        self.ego_agent_mask[valid_sampled_indices[:, 0], valid_sampled_indices[:, 1]] = True

        valid_controlled = self.controlled_agent_mask[
            valid_sampled_indices[:, 0], valid_sampled_indices[:, 1]
        ]
        all_valid = valid_controlled.all().item()
        
        if not all_valid:
            invalid_indices = valid_sampled_indices[~valid_controlled]
            print(f"Invalid indices: {invalid_indices}")

        

        self.worlds_with_agents = torch.any(self.ego_agent_mask, dim=1) 
        assert(len(self.worlds_with_agents) == self.num_worlds) 


    def assign_vehilces_to_actors_done_worlds(self,done_worlds):

        sampled_agent_indices = self.sample_vehicle_in_world(done_worlds)
        origininal_ego_mask = self.ego_agent_mask.clone()

        self.create_masks(sampled_agent_indices)

    def sample_vehicle_in_world(self, done_worlds):
            # Create a boolean mask for done worlds
            done_worlds_mask = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
            done_worlds_mask[done_worlds] = True
            sampled_indices = []
            environments = self.controllable_agent_idx[:, 0]
            
            # Only process worlds that are done
            for world in done_worlds:
                mask = environments == world
                indices_in_env = self.controllable_agent_idx[mask]
                
                # Make sure there are controllable agents in this world
                if indices_in_env.size(0) > 0:

                    random_index = torch.randint(0, indices_in_env.size(0), (1,)).item()
                    sampled_indices.append(indices_in_env[random_index])
        
            if sampled_indices:
                return torch.stack(sampled_indices)
            else:
                return torch.zeros((0, 2), dtype=self.controllable_agent_idx.dtype, device=self.device)
    
        