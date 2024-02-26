import gpudrive
import gymnasium as gym
import torch
import math
import numpy as np
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from stable_baselines3.common.env_checker import check_env

class GPUDriveEnv(gym.Env):
    def __init__(self, params: gpudrive.Parameters = None):
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            auto_reset=True,
            json_path="/home/aarav/gpudrive/build/tests/testJsons",
            params= params if params is not None else GPUDriveEnv.makeRewardParams(),
        )
        self.self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        self.partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
        self.map_obs_tensor = self.sim.map_observation_tensor().to_torch()
        self.obs_tensors, self.num_obs_features = self.setup_obs()

        # self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.num_obs_features,), dtype=torch.float32)
        self.observation_space = gym.spaces.Dict({
            "self_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[0].shape, dtype=np.float64),
            "partner_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[1].shape, dtype=np.float64),
            "map_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[2].shape, dtype=np.float64),
            "steps_remaining": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[3].shape, dtype=np.float64),
            "id": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[4].shape, dtype=np.float64),
        })

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.setup_actions(), dtype=np.float64)


    def setup_obs(self):
        self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
        map_obs_tensor = self.sim.map_observation_tensor().to_torch()
        steps_remaining_tensor = self.sim.steps_remaining_tensor().to_torch()

        N, A, O = self_obs_tensor.shape[0:3] # N = num worlds, A = num agents, O = num obs features
        batch_size = N * A

        # Add in an agent ID tensor
        id_tensor = torch.arange(A).float()
        if A > 1:
            id_tensor = id_tensor / (A - 1)

        id_tensor = id_tensor.to(device=self_obs_tensor.device)
        id_tensor = id_tensor.view(1, A).expand(N, A).reshape(batch_size, 1)

        # Flatten map obs tensor of shape (N, R, 4) to (N, 4 * R)
        map_obs_tensor = map_obs_tensor.view(N, map_obs_tensor.shape[1]*map_obs_tensor.shape[2])
        # map_obs_tensor = map_obs_tensor.view(N, 1, -1).expand(N, A, -1).reshape(batch_size, -1)
        map_obs_tensor = map_obs_tensor.repeat(N*A//N, 1)

        obs_tensors = [
            self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
            partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
            map_obs_tensor.view(batch_size, *map_obs_tensor.shape[1:]),
            steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
            id_tensor,
        ]

        num_obs_features = 0
        for tensor in obs_tensors:
            num_obs_features += math.prod(tensor.shape[1:])

        return obs_tensors, num_obs_features

    def setup_actions(self):
        action_tensor = self.sim.action_tensor().to_torch()
        return action_tensor.shape

    def step(self, action):
        action = torch.tensor(action)
        self.sim.step()
        obs, _ = self.setup_obs()
        obs = {
            "self_obs": obs[0].numpy(),
            "partner_obs": obs[1].numpy(),
            "map_obs": obs[2].numpy(),
            "steps_remaining": obs[3].numpy(),
            "id": obs[4].numpy(),
        }
        reward = torch.sum(self.sim.reward_tensor().to_torch()).numpy().astype(np.float32).item()
        done = self.sim.done_tensor().to_torch()
        info = {}
        return obs, reward, done.any().item(), False, info

    def reset(self, seed = None, options=None):
        self.sim.reset(0)
        obs, _ = self.setup_obs()
        obs = {
            "self_obs": obs[0].numpy(),
            "partner_obs": obs[1].numpy(),
            "map_obs": obs[2].numpy(),
            "steps_remaining": obs[3].numpy(),
            "id": obs[4].numpy(),
        }
        for key in obs:
            print(key, obs[key].shape)
        return obs, {}

    @staticmethod
    def makeRewardParams(rewardType: gpudrive.RewardType = gpudrive.RewardType.DistanceBased, distanceToGoalThreshold: float = 1.0, distanceToExpertThreshold: float = 1.0, polylineReductionThreshold: float = 1.0, observationRadius: float = 100.0):
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = rewardType
        reward_params.distanceToGoalThreshold = distanceToGoalThreshold
        reward_params.distanceToExpertThreshold = distanceToExpertThreshold 

        params = gpudrive.Parameters()
        params.polylineReductionThreshold = polylineReductionThreshold
        params.observationRadius = observationRadius
        params.rewardParams = reward_params  
        return params

if __name__ == "__main__":
    env = GPUDriveEnv()
    print(env.obs_tensors[0].shape)
    print(env.num_obs_features)
    env = NormalizeObservation(FlattenObservation(env))
    print("-----------------------------")
    print(env.observation_space)
    print(env.observation_space.sample())
    # model = A2C("CnnPolicy", env).learn(total_timesteps=1000)
    check_env(env, warn=True)