from pdb import set_trace as T
import os
import numpy as np
from pathlib import Path
import torch
import gymnasium

from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
    SelectionDiscipline,
)
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from pufferlib.environment import PufferEnv


def env_creator(name="gpudrive", data_dir="data/processed/examples"):
    return lambda: PufferGPUDrive(data_dir=data_dir, device="cuda")


class PufferGPUDrive(PufferEnv):
    def __init__(
        self,
        data_dir,
        device="cuda",
        max_cont_agents=32,
        num_worlds=32,
        k_unique_scenes=1,
        buf=None,
    ):
        assert buf is None, "GPUDrive set up only for --vec native"
        self.device = device
        self.data_dir = data_dir
        self.max_cont_agents = max_cont_agents
        self.num_worlds = num_worlds
        self.k_unique_scenes = k_unique_scenes
        self.total_agents = max_cont_agents * num_worlds

        # Set working directory to the base directory 'gpudrive'
        working_dir = os.path.join(Path.cwd(), "../gpudrive")
        os.chdir(working_dir)

        scene_config = SceneConfig(
            path=data_dir,
            num_scenes=num_worlds,
            discipline=SelectionDiscipline.K_UNIQUE_N,
            k_unique_scenes=k_unique_scenes,
        )

        env_config = EnvConfig()

        render_config = RenderConfig(
            draw_obj_idx=True,
        )

        self.env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            render_config=render_config,
            max_cont_agents=max_cont_agents,
            device=device,
        )

        self.obs_size = self.env.observation_space.shape[-1]
        self.single_action_space = self.env.action_space
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.float32
        )
        self.render_mode = "rgb_array"
        self.num_live = []
        self.num_agents = max_cont_agents * num_worlds

        self.controlled_agent_mask = self.env.cont_agent_mask.clone()
        self.num_controlled = self.controlled_agent_mask.sum().item()

        print(
            f"Mean number controlled agents per scene: {self.num_controlled/num_worlds}"
        )

        observations = self.env.reset()[self.controlled_agent_mask]

        # This assigns a bunch of buffers to self.
        # You can't use them because you want torch, not numpy
        # So I am careful to assign these afterwards
        super().__init__()
        self.observations = observations
        self.num_agents = observations.shape[0]

        self.masks = np.ones(self.num_agents, dtype=bool)
        self.actions = torch.zeros(
            (num_worlds, max_cont_agents), dtype=torch.int64
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

        # T()

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
                    "mean_reward_per_episode": self.episode_returns[
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

        # LOOK INTO THIS. BROKEN DATA
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

    def render(self, world_render_idx=0):
        return self.env.render(world_render_idx=world_render_idx)
