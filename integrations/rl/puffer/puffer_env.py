import os
import numpy as np
from pathlib import Path
import torch
import dataclasses
import gymnasium
import wandb
import random

from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
    SelectionDiscipline,
)
from pygpudrive.datatypes.roadgraph import LocalRoadGraphPoints

from pygpudrive.env.env_torch import GPUDriveTorchEnv

# from pygpudrive.visualize.core import plot_agent_observation
from pygpudrive.visualize.utils import img_from_fig

from pufferlib.environment import PufferEnv
from pygpudrive.datatypes.observation import (
    LocalEgoState,
)


def env_creator(
    data_dir,
    environment_config,
    train_config,
    device="cuda",
):
    return lambda: PufferGPUDrive(
        data_dir=data_dir,
        device=device,
        config=environment_config,
        train_config=train_config,
    )


class PufferGPUDrive(PufferEnv):
    """GPUDrive wrapper for PufferEnv."""

    def __init__(self, data_dir, device, config, train_config, buf=None):
        assert buf is None, "GPUDrive set up only for --vec native"

        self.data_dir = data_dir
        self.device = device
        self.config = config
        self.train_config = train_config
        self.max_cont_agents_per_env = config.max_controlled_agents
        self.num_worlds = config.num_worlds
        self.k_unique_scenes = config.k_unique_scenes
        # Total number of agents across envs, including padding
        self.total_agents = self.max_cont_agents_per_env * self.num_worlds

        # Set working directory to the base directory 'gpudrive'
        working_dir = os.path.join(Path.cwd(), "../gpudrive")
        os.chdir(working_dir)

        self.all_scene_paths = [
            os.path.join(self.data_dir, scene)
            for scene in sorted(os.listdir(self.data_dir))
            if scene.startswith("tfrecord")
        ]
        self.unique_scene_paths = list(set(self.all_scene_paths))

        scene_config = SceneConfig(
            path=data_dir,
            num_scenes=self.num_worlds,
            discipline=SelectionDiscipline.K_UNIQUE_N,
            k_unique_scenes=self.k_unique_scenes,
            seed=self.config.sampling_seed,
        )

        # Override any default environment settings
        env_config = dataclasses.replace(
            EnvConfig(),
            ego_state=config.ego_state,
            road_map_obs=config.road_map_obs,
            partner_obs=config.partner_obs,
            reward_type=config.reward_type,
            norm_obs=config.normalize_obs,
            dynamics_model=config.dynamics_model,
            collision_behavior=config.collision_behavior,
            dist_to_goal_threshold=config.dist_to_goal_threshold,
            polyline_reduction_threshold=config.polyline_reduction_threshold,
            remove_non_vehicles=config.remove_non_vehicles,
            lidar_obs=config.use_lidar_obs,
            disable_classic_obs=True if config.use_lidar_obs else False,
            obs_radius=config.obs_radius,
        )

        render_config = RenderConfig(
            draw_obj_idx=True,
        )

        self.env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            render_config=render_config,
            max_cont_agents=self.max_cont_agents_per_env,
            device=device,
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.single_action_space = self.env.action_space
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.float32
        )
        self.render_mode = "rgb_array"

        # Get the tfrecord file names for every environment
        self.dataset = self.env.dataset
        self.training_scenes_set = []
        self.env_to_files = {
            env_idx: Path(file_path).name
            for env_idx, file_path in enumerate(self.dataset)
        }

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

        # Rendering storage
        self.rendering_in_progress = {
            env_idx: False
            for env_idx in range(self.train_config.render_k_scenarios)
        }
        self.was_rendered_in_rollout = {
            env_idx: True
            for env_idx in range(self.train_config.render_k_scenarios)
        }
        self.frames = {
            env_idx: []
            for env_idx in range(self.train_config.render_k_scenarios)
        }

        self.wandb_obj = None
        self.global_step = 0
        self.iters = 0

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
        self.render() if self.train_config.render else None

        # (2) Get rewards, terminal (dones) and info
        reward = self.env.get_rewards(
            collision_weight=self.config.collision_weight,
            off_road_weight=self.config.off_road_weight,
            goal_achieved_weight=self.config.goal_achieved_weight,
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

        self.agent_episode_returns += reward
        self.episode_returns += reward_controlled
        self.episode_lengths += 1

        # (4) Use previous live agent mask as mask
        self.masks = (  # Flattend mask: (num_worlds * max_cont_agents_per_env)
            self.live_agent_mask[self.controlled_agent_mask].cpu().numpy()
        )

        # (5) Set the mask to False for _agents_ that are terminated for the next step
        # Shape: (num_worlds, max_cont_agents_per_env)
        self.live_agent_mask[terminal] = 0

        # Flatten
        terminal = terminal[self.controlled_agent_mask]

        info = []

        if len(done_worlds) > 0:

            # Log episode videos
            if self.train_config.render:
                for render_env_idx in range(
                    self.train_config.render_k_scenarios
                ):
                    if (
                        render_env_idx in done_worlds
                        and len(self.frames[render_env_idx]) > 0
                    ):
                        frames_array = np.array(self.frames[render_env_idx])
                        self.wandb_obj.log(
                            {
                                f"vis/state/env_{render_env_idx}: {self.env_to_files[render_env_idx]}": wandb.Video(
                                    np.moveaxis(frames_array, -1, 1),
                                    fps=self.train_config.render_fps,
                                    format=self.train_config.render_format,
                                    caption=f"global step: {self.global_step:,}",
                                )
                            }
                        )
                        # Reset rendering storage
                        self.frames[render_env_idx] = []
                        self.rendering_in_progress[render_env_idx] = False
                        self.was_rendered_in_rollout[render_env_idx] = True

                    # Log agent views (TODO)
                    if self.train_config.render_agent_obs:
                        first_person_views = self.render_agent_observations(
                            render_env_idx
                        )
                        # Log final observations to wandb
                        for i in range(first_person_views.shape[0]):
                            self.wandb_obj.log(
                                {
                                    f"vis/first_person_view/env_{render_env_idx}_agent_{i}": wandb.Image(
                                        first_person_views[i],
                                        caption=f"global step: {self.global_step:,}",
                                    )
                                }
                            )

            # Log episode statistics
            controlled_mask = self.controlled_agent_mask[
                done_worlds, :
            ].clone()
            info_tensor = self.env.get_infos()[done_worlds, :, :][
                controlled_mask
            ]
            agent_episode_returns = self.agent_episode_returns[done_worlds, :][
                controlled_mask
            ]

            local_roadgraph = LocalRoadGraphPoints.from_tensor(
                local_roadgraph_tensor=self.env.sim.agent_roadmap_tensor(),
                backend="torch",
                device="cuda",
            )
            rg_sparsity = (
                local_roadgraph.type[self.controlled_agent_mask] == 0
            ).sum() / local_roadgraph.type[self.controlled_agent_mask].numel()

            ego_state = LocalEgoState.from_tensor(
                self_obs_tensor=self.env.sim.self_observation_tensor(),
                backend="torch",
                device="cuda",
            )
            agent_speeds = (
                ego_state.speed[done_worlds][controlled_mask].cpu().numpy()
            )

            num_finished_agents = controlled_mask.sum().item()
            if num_finished_agents > 0:
                info.append(
                    {
                        "mean_episode_reward_per_agent": agent_episode_returns.mean().item(),
                        "perc_goal_achieved": info_tensor[:, 3].sum().item()
                        / num_finished_agents,
                        "perc_off_road": info_tensor[:, 0].sum().item()
                        / num_finished_agents,
                        "perc_veh_collisions": info_tensor[:, 1].sum().item()
                        / num_finished_agents,
                        "perc_non_veh_collision": info_tensor[:, 2]
                        .sum()
                        .item()
                        / num_finished_agents,
                        "control_density": (
                            self.num_agents
                            / self.controlled_agent_mask.numel()
                        ),
                        "total_controlled_agents": self.num_agents,
                        "episode_length": self.episode_lengths[done_worlds, :]
                        .mean()
                        .item(),
                        "rg_sparsity": rg_sparsity.item(),
                        "mean_agent_speed": agent_speeds.mean().item(),
                    }
                )

            # Asynchronously reset the done worlds and empty storage
            for idx in done_worlds:
                self.env.sim.reset([idx])
                self.episode_returns[idx] = 0
                self.agent_episode_returns[idx, :] = 0
                self.episode_lengths[idx, :] = 0
                # Reset the live agent mask so that the next alive mask will mark
                # all agents as alive for the next step
                self.live_agent_mask[idx] = self.controlled_agent_mask[idx]

        # (6) Get the next observations. Note that we do this after resetting
        # the worlds so that we always return a fresh observation
        next_obs = self.env.get_obs()[self.controlled_agent_mask]

        self.observations = next_obs
        self.rewards = reward_controlled
        self.terminals = terminal
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        """Render the environment based on conditions.
        - If the episode has just started, start a new rendering.
        - If the episode is in progress, continue rendering.
        - If the episode has ended, log the video to WandB.
        - Only render env once per rollout.
        """
        for render_env_idx in range(self.train_config.render_k_scenarios):
            # Start a new rendering if the episode has just started
            if self.iters % self.train_config.render_interval == 0:
                if (
                    self.episode_lengths[render_env_idx, :][0] == 0
                    and not self.was_rendered_in_rollout[render_env_idx]
                ):
                    self.rendering_in_progress[render_env_idx] = True

        # Continue rendering if in progress
        if self.train_config.render_simulator_state:
            envs_to_render = list(
                np.where(np.array(list(self.rendering_in_progress.values())))[
                    0
                ]
            )
            time_steps = list(self.episode_lengths[envs_to_render, 0])

            sim_state_figures = self.vis.plot_simulator_state(
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
            observation_fig, _ = self.vis.plot_agent_observation(
                env_idx=env_idx,
                agent_idx=agent_id.item(),
            )

            img_arrays.append(img_from_fig(observation_fig))

        return np.array(img_arrays)

    def resample_scenario_batch(self):
        """Sample a new set of WOMD scenarios."""
        if self.train_config.resample_mode == "random":
            total_unique = len(self.unique_scene_paths)

            # Update set of scenes we've trained on
            self.training_scenes_set.append(set(self.dataset))

            # Reset
            self.dataset = []

            # Sample batch of unique scenes
            if self.num_worlds <= total_unique:
                self.dataset = random.sample(
                    self.unique_scene_paths, self.num_worlds
                )
            else:
                # If N is greater, repeat the unique scenes until we get N scenes
                while len(self.dataset) < self.num_worlds:
                    self.dataset.extend(
                        random.sample(self.unique_scene_paths, total_unique)
                    )
                    if len(self.dataset) > self.num_worlds:
                        self.dataset = self.dataset[: self.num_worlds]
        else:
            raise NotImplementedError(
                f"Resample mode {self.train_config.resample_mode} is currently not supported."
            )

        # Re-initialize the simulator with the new dataset
        print(
            f"Re-initializing sim with {len(set(self.dataset))} {self.train_config.resample_mode} unique scenes.\n"
        )

        self.env.reinit_scenarios(self.dataset)

        # Update controlled agent mask and other masks
        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.num_agents = self.controlled_agent_mask.sum().item()
        self.masks = np.ones(self.num_agents, dtype=bool)
        self.agent_ids = np.arange(self.num_agents)
        print(f"Total controlled agents across worlds: {self.num_agents}")

        self.reset()  # Reset storage
        # Get info from new worlds
        self.observations = self.env.reset()[self.controlled_agent_mask]

        # Get the tfrecord file names for every environment
        self.env_to_files = {
            env_idx: Path(file_path).name
            for env_idx, file_path in enumerate(self.dataset)
        }

    def clear_render_storage(self):
        """Clear rendering storage."""
        for env_idx in range(self.train_config.render_k_scenarios):
            self.frames[env_idx] = []
            self.rendering_in_progress[env_idx] = False
            self.was_rendered_in_rollout[env_idx] = False
            