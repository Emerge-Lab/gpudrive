import os
import numpy as np
from pathlib import Path
import torch
import dataclasses
import gymnasium

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
    device="cuda",
):
    return lambda: PufferGPUDrive(
        data_dir=data_dir,
        device=device,
        config=environment_config,
    )


class PufferGPUDrive(PufferEnv):
    """GPUDrive wrapper for PufferEnv."""

    def __init__(self, data_dir, device, config, buf=None):
        assert buf is None, "GPUDrive set up only for --vec native"

        self.data_dir = data_dir
        self.device = device
        self.config = config
        self.max_cont_agents = config.max_controlled_agents
        self.num_worlds = config.num_worlds
        self.k_unique_scenes = config.k_unique_scenes
        self.total_agents = self.max_cont_agents * self.num_worlds

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
        )

        render_config = RenderConfig(
            draw_obj_idx=True,
        )

        self.env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            render_config=render_config,
            max_cont_agents=self.max_cont_agents,
            device=device,
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.single_action_space = self.env.action_space
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.float32
        )
        self.render_mode = "rgb_array"
        self.num_live = []
        self.num_agents = self.max_cont_agents * self.num_worlds

        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.num_controlled = self.controlled_agent_mask.sum().item()

        observations = self.env.reset()[self.controlled_agent_mask]

        # This assigns a bunch of buffers to self.
        # You can't use them because you want torch, not numpy
        # So I am careful to assign these afterwards
        super().__init__()
        self.observations = observations
        self.num_agents = observations.shape[0]

        self.masks = np.ones(self.num_agents, dtype=bool)
        self.actions = torch.zeros(
            (self.num_worlds, self.max_cont_agents), dtype=torch.int64
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

        self.episode_returns = torch.zeros(
            self.num_agents, dtype=torch.float32
        ).to(self.device)
        self.episode_lengths = torch.zeros(
            self.num_agents, dtype=torch.float32
        ).to(self.device)
        self.live_agent_mask = torch.ones(
            (self.num_worlds, self.max_cont_agents), dtype=bool
        ).to(self.device)

        self.observations[self.observations > 10] = 0
        self.observations[self.observations < 10] = 0

        return self.observations, []

    def step(self, action):
        action = torch.from_numpy(action).to(self.device)
        self.actions[self.controlled_agent_mask] = action
        self.env.step_dynamics(self.actions)

        obs = self.env.get_obs()[self.controlled_agent_mask]
        reward = self.env.get_rewards()[self.controlled_agent_mask]
        terminal = self.env.get_dones().bool()

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

        self.episode_returns += reward
        self.episode_lengths += 1
        self.masks = (
            self.live_agent_mask[self.controlled_agent_mask].cpu().numpy()
        )
        self.live_agent_mask[terminal] = 0
        terminal = terminal[self.controlled_agent_mask]

        info = []
        self.num_live.append(self.masks.sum())

        if len(done_worlds) > 0:
            controlled_mask = self.controlled_agent_mask[done_worlds]
            info_tensor = self.env.get_infos()[done_worlds][controlled_mask]
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
                    "mean_episode_reward_per_agent": self.episode_returns[
                        done_worlds
                    ]
                    .mean()
                    .item(),
                    "control_density": self.num_controlled / self.num_agents,
                    "alive_density": np.mean(self.num_live) / self.num_agents,
                    "num_finished_agents": num_finished_agents,
                    "episode_length": self.episode_lengths[done_worlds]
                    .mean()
                    .item(),
                }
            )

            self.num_live = []
            for idx in done_worlds:
                self.env.sim.reset([idx])
                self.episode_returns[idx] = 0
                self.episode_lengths[idx] = 0
                self.live_agent_mask[idx] = self.controlled_agent_mask[idx]

        # TODO: Look in
        obs[obs > 10] = 0
        obs[obs < -10] = 0

        self.observations = obs
        self.rewards = reward
        self.terminals = terminal
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render_agent_observations(self, env_idx, time_step):
        """Render a single observation."""
        agent_ids = torch.where(self.controlled_agent_mask[env_idx, :])[
            0
        ].cpu()

        ego_state = LocalEgoState.from_tensor(
            self_obs_tensor=self.env.sim.self_observation_tensor(),
            backend=self.env.backend,
            device="cpu",
        )

        local_roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=self.env.sim.agent_roadmap_tensor(),
            backend=self.env.backend,
            device="cpu",
        )

        partner_obs = PartnerObs.from_tensor(
            partner_obs_tensor=self.env.sim.partner_observations_tensor(),
            backend=self.env.backend,
            device="cpu",
        )

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
