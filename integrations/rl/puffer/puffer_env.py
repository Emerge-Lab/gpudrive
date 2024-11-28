import os
import numpy as np
from pathlib import Path
import torch
import dataclasses
import gymnasium
import wandb

from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
    SelectionDiscipline,
)
from pygpudrive.datatypes.observation import (
    LocalEgoState,
    PartnerObs,
    LidarObs,
)
from pygpudrive.datatypes.roadgraph import LocalRoadGraphPoints

from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.visualize.core import plot_agent_observation
from pygpudrive.visualize.utils import img_from_fig

from pufferlib.environment import PufferEnv


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

        # Initialize rendering buffers
        self.clear_render_buffer()

        # Set working directory to the base directory 'gpudrive'
        working_dir = os.path.join(Path.cwd(), "../gpudrive")
        os.chdir(working_dir)

        scene_config = SceneConfig(
            path=data_dir,
            num_scenes=self.num_worlds,
            discipline=SelectionDiscipline.K_UNIQUE_N,
            k_unique_scenes=self.k_unique_scenes,
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
        self.num_live = []

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
        self.render_terminals = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env), dtype=torch.bool
        ).to(self.device)
        self.episode_returns = torch.zeros(
            self.num_agents, dtype=torch.float32
        ).to(self.device)
        self.agent_episode_lengths = torch.zeros(
            self.num_agents, dtype=torch.float32
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

        # (2) Get rewards, terminal (dones) and info
        reward = self.env.get_rewards()[self.controlled_agent_mask]
        terminal = self.env.get_dones().bool()

        self.episode_returns += reward
        self.agent_episode_lengths += 1
        self.episode_lengths += 1

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

        # (4) Use previous live agent mask as mask
        self.masks = (  # Flattend mask: (num_worlds * max_cont_agents_per_env)
            self.live_agent_mask[self.controlled_agent_mask].cpu().numpy()
        )

        # (5) Set the mask to False for _agents_ that are terminated for the next step
        # Shape: (num_worlds, max_cont_agents_per_env)
        self.live_agent_mask[terminal] = 0

        info = []
        self.num_live.append(self.masks.sum())

        if len(done_worlds) > 0:

            local_roadgraph = LocalRoadGraphPoints.from_tensor(
                local_roadgraph_tensor=self.env.sim.agent_roadmap_tensor(),
                backend="torch",
                device="cuda",
            )
            rg_sparsity = (
                local_roadgraph.type[self.controlled_agent_mask] == 0
            ).sum() / local_roadgraph.type[self.controlled_agent_mask].numel()

            # Log episode statistics
            controlled_mask = self.controlled_agent_mask[
                done_worlds, :
            ].clone()
            info_tensor = self.env.get_infos()[done_worlds, :, :][
                controlled_mask
            ]
            num_finished_agents = controlled_mask.sum().item()
            info.append(
                {
                    "perc_off_road": info_tensor[:, 0].sum().item()
                    / num_finished_agents,
                    "perc_veh_collisions": info_tensor[:, 1].sum().item()
                    / num_finished_agents,
                    "perc_non_veh_collision": info_tensor[:, 2].sum().item()
                    / num_finished_agents,
                    "perc_goal_achieved": info_tensor[:, 3].sum().item()
                    / num_finished_agents,
                    "sum_goal_achieved": info_tensor[:, 3].sum().item(),
                    "num_controlled_agents": num_finished_agents,
                    # TODO: Bug. This needs to be indexed differently
                    # "mean_episode_reward_per_agent": self.episode_returns[
                    #     done_worlds
                    # ]
                    # .mean()
                    # .item(),
                    "control_density": (
                        self.num_agents / self.controlled_agent_mask.numel()
                    ),
                    "alive_density": np.mean(self.num_live) / self.num_agents,
                    # TODO: Bug. This needs to be indexed differently
                    "episode_length": self.episode_lengths[done_worlds, :]
                    .mean()
                    .item(),
                    "rg_sparsity": rg_sparsity.item(),
                }
            )
            self.num_live = []

            # Asynchronously reset the done worlds and empty storage
            for idx in done_worlds:
                self.env.sim.reset([idx])
                self.episode_returns[idx] = 0
                # TODO: Bug. This needs to be indexed differently
                self.episode_lengths[idx] = 0
                # Reset the live agent mask so that the next alive mask will mark
                # all agents as alive for the next step
                self.live_agent_mask[idx] = self.controlled_agent_mask[idx]
                # Reset terminals for reset envs
                # terminal[idx, :] = self.env.get_dones()[idx, :].bool()

        # Flatten
        flat_terminal = terminal[self.controlled_agent_mask]

        # (6) Get the next observations. Note that we do this after resetting
        # the worlds so that we always return a fresh observation
        next_obs = self.env.get_obs()[self.controlled_agent_mask]

        self.observations = next_obs
        self.rewards = reward
        self.terminals = flat_terminal
        self.render_terminals = terminal
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self, wandb_obj=None, global_step=None):
        """Render logic for making videos of agent behaviors during rollout.

        Args:
            env_idx (_type_): _description_
        """

        # Check if render buffers are not full
        if not all(self.already_rendered_during_rollout.values()):

            for env_idx in range(self.train_config.render_k_scenarios):
                if (
                    all(self.episode_lengths[0, :]) == 0
                    and not self.already_rendered_during_rollout[
                        f"env_{env_idx}"
                    ]
                ):
                    self.rendering_in_process[f"env_{env_idx}"] = True
                    print(f"start env_{env_idx}...")

                if self.rendering_in_process[f"env_{env_idx}"]:
                    # Render simulator state
                    if self.train_config.render_simulator_state:
                        print("Rendering simulator state")
                        simulator_state = self.render_simulator_state(env_idx)
                        self.simulator_state_dict[f"env_{env_idx}"].append(
                            simulator_state
                        )
                    # Render agent observations
                    if self.train_config.render_agent_obs:
                        print("Rendering agent observations")
                        agent_observations = self.render_agent_observations(
                            env_idx=env_idx,
                            time_step=int(
                                int(self.episode_lengths[env_idx, :][0])
                            ),
                        )
                        self.agent_frames_dict[f"env_{env_idx}"].append(
                            agent_observations
                        )

                if (
                    all(self.render_terminals[env_idx, :])
                    and self.rendering_in_process[f"env_{env_idx}"]
                ):
                    # Stop if episode has ended and render
                    print(f"end rendering env_{env_idx}...")
                    self.rendering_in_process[f"env_{env_idx}"] = False
                    self.already_rendered_during_rollout[
                        f"env_{env_idx}"
                    ] = True

                    # Render simulator state
                    if self.train_config.render_simulator_state:
                        frames_array = np.array(
                            self.simulator_state_dict[f"env_{env_idx}"]
                        )
                        wandb_obj.log(
                            {
                                f"Video/State/env_{env_idx}": wandb.Video(
                                    np.moveaxis(frames_array, -1, 1),
                                    fps=self.train_config.render_fps,
                                    format=self.train_config.render_format,
                                    caption=f"global step: {global_step:,}",
                                )
                            }
                        )
                        del self.simulator_state_dict[f"env_{env_idx}"][:]

                    if self.train_config.render_agent_obs:
                        print(
                            f"Rendering agent observations for env_{env_idx}"
                        )
                        num_agents = self.agent_frames_dict[f"env_{env_idx}"][
                            0
                        ].shape[0]
                        # Render agent observations
                        for agent_idx in range(num_agents):
                            # Stack the frames for this agent along time axis
                            agent_frames = np.stack(
                                [
                                    image[agent_idx]
                                    for image in self.agent_frames_dict[
                                        f"env_{env_idx}"
                                    ]
                                ],
                                axis=0,
                            )
                            agent_frames = np.transpose(
                                agent_frames, (0, 3, 1, 2)
                            )
                            # Shape: (time_steps, img_width, img_height, 3)
                            wandb_obj.log(
                                {
                                    f"Video/Observation/env_{env_idx}_agent_{agent_idx}": wandb.Video(
                                        agent_frames,
                                        fps=self.train_config.render_fps,
                                        format=self.train_config.render_format,
                                        caption=f"global step: {global_step:,}",
                                    )
                                }
                            )
                        del self.agent_frames_dict[f"env_{env_idx}"][:]

    def render_agent_observations(self, env_idx, time_step):
        """Render a single observation."""
        agent_ids = torch.where(self.controlled_agent_mask[env_idx, :])[
            0
        ].cpu()

        ego_state = LocalEgoState.from_tensor(
            self_obs_tensor=self.env.sim.self_observation_tensor(),
            backend=self.env.backend,
            device="cpu",
        ).clone()

        local_roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=self.env.sim.agent_roadmap_tensor(),
            backend=self.env.backend,
            device="cpu",
        ).clone()

        partner_obs = PartnerObs.from_tensor(
            partner_obs_tensor=self.env.sim.partner_observations_tensor(),
            backend=self.env.backend,
            device="cpu",
        ).clone()

        img_arrays = []

        for agent_id in agent_ids:

            observation_fig, _ = plot_agent_observation(
                env_idx=env_idx,
                agent_idx=agent_id.item(),
                observation_roadgraph=local_roadgraph,
                observation_ego=ego_state,
                observation_partner=partner_obs,
                time_step=time_step,
            )

            img_arrays.append(img_from_fig(observation_fig))

        return np.array(img_arrays)

    def render_simulator_state(self, env_idx):
        """Render the simulator state from a bird's eye view."""
        simulator_state_array = self.env.render(env_idx)
        return simulator_state_array

    def clear_render_buffer(self):
        """Set rendering buffers to empty."""
        # Rendering storage
        self.already_rendered_during_rollout = {
            f"env_{i}": False
            for i in range(self.train_config.render_k_scenarios)
        }
        self.rendering_in_process = {
            f"env_{i}": False
            for i in range(self.train_config.render_k_scenarios)
        }
        self.agent_frames_dict = {
            f"env_{i}": [] for i in range(self.train_config.render_k_scenarios)
        }
        self.simulator_state_dict = {
            f"env_{i}": [] for i in range(self.train_config.render_k_scenarios)
        }
