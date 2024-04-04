import gpudrive
import gymnasium as gym
import torch
import torch.nn as nn
import math
import numpy as np
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from stable_baselines3.common.env_checker import check_env
from sim_utils.creator import SimCreator
import pufferlib.models

class GPUDriveEnv(gym.Env):
    Recurrent = None

    def __init__(self, params: gpudrive.Parameters = None):
        self.sim = SimCreator()
        self.self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        self.partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
        self.map_obs_tensor = self.sim.map_observation_tensor().to_torch()
        self.shape_tensor = self.sim.shape_tensor().to_torch()
        self.obs_tensors, self.num_obs_features = self.setup_obs()



        self.num_envs = self.obs_tensors[0].shape[0]
        self.agents_per_env = self.shape_tensor[:,0]
        self.total_num_agents = torch.sum(self.agents_per_env)

        self.single_observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.num_obs_features], dtype=np.float64)
        # self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.num_obs_features,), dtype=torch.float32)
        self.observation_space = gym.spaces.Dict({
            "self_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[0].shape, dtype=np.float64),
            "partner_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[1].shape, dtype=np.float64),
            "map_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[2].shape, dtype=np.float64),
            "steps_remaining": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[3].shape, dtype=np.float64),
            "id": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[4].shape, dtype=np.float64),
        })

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.setup_actions(), dtype=np.float64)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=self.setup_actions()[2:], dtype=np.float64)

        self.Recurrent = None
        self.unflatten_context = None
        self.mask_agents = True

    def filter_padding_agents(self, obs_tensors, N, A):
        # Calculate the batch size assuming it's N*A for simplicity here
        batch_size = N * A
        
        # Generate a mask for valid agents across all environments
        valid_counts = self.shape_tensor[:, 0]  # Shape: (N,)
        cumulative_counts = torch.arange(A).expand(N, A)  # Matrix of shape (N, A) with repeated range(A)
        valid_mask = cumulative_counts < valid_counts.unsqueeze(1)  # Expanding valid_counts to match dimensions

        # Flatten the mask to match the batched observation shape
        flat_valid_mask = valid_mask.view(-1)
        
        # Filter observations for each tensor in obs_tensors
        filtered_obs_tensors = [tensor.view(batch_size, -1)[flat_valid_mask].view(-1, tensor.shape[-1]) for tensor in obs_tensors]

        return filtered_obs_tensors
    
    def apply_action(self, actions):
        N, A = self.self_obs_tensor.shape[0:2]
        batch_size = N * A
        action_dim = actions.shape[1]
        
        # Create a placeholder tensor for the actions, filled with zeros or an appropriate default value
        # You may need to adjust the default value based on the nature of your actions
        padded_actions = torch.zeros(batch_size, action_dim, device=actions.device)
        
        # Generate the mask for valid agents, similar to the filtering process
        valid_counts = self.shape_tensor[:, 0]  # Valid agent counts per environment
        cumulative_counts = torch.arange(A).expand(N, A)  # Matrix of shape (N, A) with repeated range(A)
        valid_mask = cumulative_counts < valid_counts.unsqueeze(1)  # Mask for valid agents
        
        # Flatten the mask to match the batched action shape
        flat_valid_mask = valid_mask.view(-1)
        
        # Compute indices where the valid actions should be inserted
        valid_indices = flat_valid_mask.nonzero(as_tuple=True)[0]
        
        # Insert the filtered actions into the padded actions tensor
        padded_actions[valid_indices] = actions

        # Reshape padded_actions to match the expected shape for the environment, if necessary
        # For example, if your environment expects a specific shape, you might need to adjust it accordingly

        return padded_actions

    def setup_obs(self):
        self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
        # map_obs_tensor = self.sim.map_observation_tensor().to_torch()
        agent_map_obs_tensor = self.sim.agent_roadmap_tensor().to_torch()
        controlled_state_tensor = self.sim.controlled_state_tensor().to_torch()
        done_tensor = self.sim.done_tensor().to_torch()
        reward_tensor = self.sim.reward_tensor().to_torch()
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
        # map_obs_tensor = map_obs_tensor.view(N, map_obs_tensor.shape[1]*map_obs_tensor.shape[2])
        map_obs_tensor = agent_map_obs_tensor.view(N, A, -1)
        partner_obs_tensor = partner_obs_tensor.view(N, A, -1)
        # map_obs_tensor = map_obs_tensor.repeat(N*A//N, 1)

        obs_tensors = [
            self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
            partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
            map_obs_tensor.view(batch_size, *map_obs_tensor.shape[2:]),
            controlled_state_tensor.view(batch_size, *controlled_state_tensor.shape[2:]),
            done_tensor.view(batch_size, *done_tensor.shape[2:]),
            reward_tensor.view(batch_size, *reward_tensor.shape[2:]),
            steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
            id_tensor,
        ]
        obs_tensors = self.filter_padding_agents(obs_tensors, N, A)

        num_obs_features = 0
        for tensor in obs_tensors[:3]:
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
    
    def async_reset(self, seed = None):
        # seed is ignored. The sim is deterministic.
        self.data = [self.reset(i) for i in range(self.num_envs)]

    def recv(self):
        recvs = []
        next_env_id = []
        obs, _ = self.setup_obs()
        env_obs = torch.cat(obs[:3], dim=-1)
        controlled, dones, rews= obs[3], obs[4], obs[5]
        truncateds = torch.Tensor([False] * self.total_num_agents).to(dones.device)
        infos = [{} for _ in range(self.total_num_agents)]

        infos = [i for ii in infos for i in ii]
        
        mask = torch.logical_not(torch.logical_or(dones.bool(), controlled.bool()))

        env_ids = torch.arange(self.total_num_agents).to(dones.device)
        env_ids = env_ids//self.total_num_agents

        if self.mask_agents:
            return env_obs, rews, dones, truncateds, infos, env_ids, mask

        return env_obs, rews, dones, truncateds, infos, env_ids

    def send(self, actions):
        actions = self.apply_action(actions)
        actions = actions.view(self.self_obs_tensor.shape[0], -1, self.single_action_space.shape[0])
        self.sim.action_tensor().to_torch().copy_(actions)
        self.sim.step()

    def reset(self, env_id, seed = None, options=None):
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
    
    @staticmethod
    def Policy(env: gym.Env):
        return torch.rand(1)


class Convolutional1D(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.num_actions = env.single_action_space.shape[0]
        self.channels_last = channels_last
        self.downsample = downsample
        
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv1d(framestack, 32, 128, stride=32)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv1d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv1d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1024),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(64*1024, hidden_size)),
            nn.ReLU(),
        )
        # Discrete
        # self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

        # continuous
        # Output layers for the mean and log_std of the actions
        self.mean = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, self.num_actions), std=0.01)
        # It's common to initialize the log_std as a learnable parameter rather than an output of the network
        # This can help with stability and is often sufficient for many environments
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        observations = observations.unsqueeze(1)  # This adds a channel dimension, resulting in [batch_size, 1, length]
        return self.network(observations.float()), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        hidden, lookup = self.encode_observations(env_outputs)
        # Discrete
        # actions, value = self.decode_actions(hidden, lookup)

        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(log_std)
        value = self.value_fn(hidden)
        # return actions, value
        return [mean, std], value

class Policy(pufferlib.models.Policy):
    def __init__(self, env):
        input_size=512
        hidden_size=512
        output_size=512
        framestack=1
        flat_size=64*7*7
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
        )

def make_gpudrive():
    return GPUDriveEnv()

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