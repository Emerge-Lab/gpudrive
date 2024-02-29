from dataclasses import dataclass
import glob
import math
import time

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch

import gpudrive
from logger import MetricsLogger


@dataclass
class StepReturn:
    """Return type for step() method"""

    state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    valids: torch.Tensor

class Env:
    def __init__(self):
        # Create an instance of RewardParams
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = gpudrive.RewardType.DistanceBased  # Or any other value from the enum
        reward_params.distanceToGoalThreshold = 1.0  # Set appropriate values
        reward_params.distanceToExpertThreshold = 1.0  # Set appropriate values

        # Create an instance of Parameters
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = 0.5  # Set appropriate value
        params.observationRadius = 10.0  # Set appropriate value
        params.rewardParams = reward_params  # Set the rewardParams attribute to the instance created above

        data_dir = "waymo_data"
        NUM_WORLDS = len(glob.glob(f"{data_dir}/*.json"))
        self.num_sims = NUM_WORLDS
        device = 'cuda'
        # Now use the 'params' instance when creating SimManager
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CPU if device == 'cpu' else gpudrive.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=NUM_WORLDS,
            auto_reset=True,
            json_path="waymo_data",
            params=params
        )
        self.num_agents = self.sim.shape_tensor().to_torch()[0, 0]
        # TODO(ev) remove hardcoding
        self.steer_actions = torch.tensor([-0.6, 0, 0.6], device=device)
        self.accel_actions = torch.tensor([-3, 0, 3], device=device)
        # self.head_actions = torch.tensor([0], device=device)
        # create a tensor of all possible actions that has dimensions
        # (N * A, self.steer_actions.shape[0] * self.accel_actions.shape[0], 2)
        # TODO(ev) don't loop like this
        self.actions = torch.zeros((self.steer_actions.shape[0] * self.accel_actions.shape[0], 2), device=device)
        idx = 0
        for steer in self.steer_actions:
            for accel in self.accel_actions:
                # TODO(ev) remove head angle harcoding
                self.actions[idx] = torch.tensor([steer, accel])
                idx += 1
        # TODO(ev) this will break when there is more than 1 world but it'll do for now
        self.actions = self.actions.unsqueeze(0).expand(NUM_WORLDS * self.sim.shape_tensor().to_torch()[0, 0], -1, -1) 
        # TODO(ev) remove this fake thing
        self.head_action = torch.zeros((self.num_sims, self.num_agents, 1), device=device)   
        
        cfg = {"num_sims": self.num_sims, "num_agents": self.num_agents, "device": device}
        self.metrics = MetricsLogger(cfg)
        
    def setup_actions(self):
        action_tensor = self.sim.action_tensor().to_torch()
        return action_tensor.shape
        
    @property
    def action_space(self):
        # TODO(ev) hardcoding
        return Discrete(n=int(self.actions.shape[1]))

    @property
    def observation_space(self):
        # TODO(ev) make this configurable
        # self.observation_space = gym.spaces.Dict({
        #     "self_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[0].shape, dtype=np.float64),
        #     "partner_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[1].shape, dtype=np.float64),
        #     "map_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[2].shape, dtype=np.float64),
        #     "steps_remaining": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[3].shape, dtype=np.float64),
        #     "id": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[4].shape, dtype=np.float64),
        # })
        return Box(low=-np.inf, high=np.inf, shape=(self.get_obs().shape[-1],))
    
    def get_obs(self):
        # TODO(ev) remove magiv numbers
        self_obs_tensor = self.sim.self_observation_tensor().to_torch()[:, :, 2:]
        partner_obs_tensor = self.sim.partner_observations_tensor().to_torch().flatten(start_dim=2)
        map_obs_tensor = self.sim.map_observation_tensor().to_torch().flatten(start_dim=1).unsqueeze(1).repeat((1, self_obs_tensor.shape[1], 1))
        return torch.cat([self_obs_tensor, partner_obs_tensor, map_obs_tensor], dim=-1)

    def reset(self):
        for i in range(self.num_sims):
            self.sim.reset(i)
        return self.get_obs()
    
    def step(self, action):
        gather_action = torch.gather(self.actions, 1, action.long().unsqueeze(-1).expand(-1, 2).unsqueeze(1)).reshape(
            (self.num_sims, self.num_agents, -1)
        )
        self.sim.action_tensor().to_torch().copy_(torch.cat((gather_action, self.head_action), dim=-1))
        self.sim.step()
        state = self.get_obs()
        reward = self.sim.reward_tensor().to_torch()
        done = self.sim.done_tensor().to_torch()
        valids = torch.logical_not(self.sim.collision_tensor().to_torch())
        import ipdb; ipdb.set_trace()
        # TODO(ev) find all the rows where valids is False in every column
        episode_over = valids.sum(dim=1) == 0
        for world_index in torch.where(episode_over)[0]:
            self.sim.reset(world_index)
        
        self.metrics.step(StepReturn(state, reward.squeeze(dim=-1), done.squeeze(dim=-1), valids.squeeze(dim=-1)))
        return state, reward, done, valids
    
if __name__ == '__main__':
    env = Env()
    env.reset()
    env.step(torch.Tensor([1,2,3]))