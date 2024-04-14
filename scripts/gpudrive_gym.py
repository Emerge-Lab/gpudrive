import gpudrive
import gymnasium as gym

from gymnasium.spaces import Box, Discrete
import pufferlib.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from stable_baselines3.common.env_checker import check_env
from sim_utils.creator import SimCreator
import pufferlib.models
import imageio
from itertools import product

class GPUDriveEnv(gym.Env):
    Recurrent = None

    def __init__(self, params: gpudrive.Parameters = None, action_space_type: str = "continuous"):
        self.sim = SimCreator()
        self.self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        self.partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
        self.map_obs_tensor = self.sim.map_observation_tensor().to_torch()
        self.shape_tensor = self.sim.shape_tensor().to_torch()
        self.obs_tensors, self.num_obs_features = self.setup_obs()

        self.action_space_type = action_space_type

        if self.action_space_type == "discrete":
            self.action_space = self.setup_discrete_actions()
            self.single_action_space = self.action_space
        elif self.action_space_type == "continuous":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.setup_actions(), dtype=np.float64)
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=self.setup_actions()[2:], dtype=np.float64)

        self.num_envs = self.self_obs_tensor.shape[0]
        self.agents_per_env = self.shape_tensor[:,0]
        self.total_num_agents = torch.sum(self.agents_per_env)
        self.env_ids = torch.repeat_interleave(torch.arange(self.num_envs).to(self.self_obs_tensor.device), self.agents_per_env)

        self.single_observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.num_obs_features], dtype=np.float64)
        # self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.num_obs_features,), dtype=torch.float32)
        self.observation_space = gym.spaces.Dict({
            "self_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[0].shape, dtype=np.float64),
            "partner_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[1].shape, dtype=np.float64),
            "map_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[2].shape, dtype=np.float64),
            "steps_remaining": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[3].shape, dtype=np.float64),
            "id": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[4].shape, dtype=np.float64),
        })



        self.Recurrent = None
        self.unflatten_context = None
        self.mask_agents = True

    def setup_discrete_actions(self):
        """Configure the discrete action space."""

        self.steer_actions = torch.tensor([-0.6, 0, 0.6])
        self.accel_actions = torch.tensor([-3, -1, 0.5, 0, 0.5, 1, 3])
        self.head_actions = torch.tensor([0])

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}

        for action_idx, (accel, steer, head) in enumerate(
            product(self.accel_actions, self.steer_actions, self.head_actions)
        ):
            self.action_key_to_values[action_idx] = [
                accel.item(),
                steer.item(),
                head.item(),
            ]

        return Discrete(n=int(len(self.action_key_to_values)))

    def filter_padding_agents(self, obs_tensors, N, A):
        # Calculate the batch size assuming it's N*A for simplicity here
        batch_size = N * A
        
        # Generate a mask for valid agents across all environments
        valid_counts = self.shape_tensor[:, 0]  # Shape: (N,)
        cumulative_counts = torch.arange(A).expand(N, A).to(obs_tensors[0].device)  # Matrix of shape (N, A) with repeated range(A)
        valid_mask = cumulative_counts < valid_counts.unsqueeze(1)  # Expanding valid_counts to match dimensions

        # Flatten the mask to match the batched observation shape
        flat_valid_mask = valid_mask.view(-1)
        
        # Filter observations for each tensor in obs_tensors
        filtered_obs_tensors = [tensor.view(batch_size, -1)[flat_valid_mask].view(-1, tensor.shape[-1]) for tensor in obs_tensors]

        return filtered_obs_tensors
    
    def apply_discrete_action(self, actions):
        # Convert the discrete action indices to action values
        actions = actions.view(-1)
        action_value_tensor = torch.zeros(actions.shape[0], 3)
        for idx, action in enumerate(actions):
            action_idx = action.item()
            action_value_tensor[idx, :] = torch.Tensor(
                    self.action_key_to_values[action_idx]
                )
        action_value_tensor.to(actions.device)
        return action_value_tensor
    
    def apply_action(self, actions):
        if self.action_space_type == "discrete":
            actions = self.apply_discrete_action(actions)

        N, A = self.self_obs_tensor.shape[0:2]
        batch_size = N * A
        action_dim = actions.shape[1]
        
        # Create a placeholder tensor for the actions, filled with zeros or an appropriate default value
        # You may need to adjust the default value based on the nature of your actions
        
        padded_actions = torch.zeros(batch_size, action_dim, device=actions.device)

        
        # Generate the mask for valid agents, similar to the filtering process
        valid_counts = self.shape_tensor[:, 0]  # Valid agent counts per environment
        cumulative_counts = torch.arange(A).expand(N, A).to(self.self_obs_tensor.device)  # Matrix of shape (N, A) with repeated range(A)
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
    
    def render(self):
        rgb_image = self.sim.rgb_tensor().to_torch()
        return rgb_image

    def render_depth(self):
        depth_image = self.sim.depth_tensor().to_torch()
        return depth_image

    def setup_obs(self):
        self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
        # map_obs_tensor = self.sim.map_observation_tensor().to_torch()
        agent_map_obs_tensor = self.sim.agent_roadmap_tensor().to_torch()
        controlled_state_tensor = self.sim.controlled_state_tensor().to_torch()
        done_tensor = self.sim.done_tensor().to_torch()
        reward_tensor = self.sim.reward_tensor().to_torch()
        steps_remaining_tensor = self.sim.steps_remaining_tensor().to_torch()

        controlled_mask = controlled_state_tensor[:, :, 0] == 1
        for i in range(done_tensor.shape[0]):
            if(done_tensor[i].all()):
                # print("done due to time")
                self.reset(i)
            elif(done_tensor[i][controlled_mask[i]].all()):
                # print("done due to agent")
                self.reset(i)

        # Add L2 Norm to obs_tensor for each agent at indexs [2,3]

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
        # print("map_obs_tensor", map_obs_tensor.shape)
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
        for tensor in obs_tensors[:1]:
            num_obs_features += math.prod(tensor.shape[1:])

        # print("num_obs_features", num_obs_features)

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
        env_obs = torch.cat(obs[:1], dim=-1)
        controlled, dones, rews= obs[3], obs[4], obs[5]
        truncateds = torch.Tensor([False] * self.total_num_agents).to(dones.device)
        infos = [{} for _ in range(self.total_num_agents)]

        infos = [i for ii in infos for i in ii]
        
        mask = controlled.bool()

        env_ids = torch.arange(self.total_num_agents).to(dones.device)
        env_ids = env_ids//self.total_num_agents

        assert(torch.isnan(env_obs).any() == False)
        if self.mask_agents:
            # iterate through environments and get all controlled agents
                
            return env_obs, rews, dones, truncateds, infos, env_ids, mask

        return env_obs, rews, dones, truncateds, infos, env_ids

    def send(self, actions):
        actions = self.apply_action(actions)
        actions = actions.view(self.self_obs_tensor.shape[0], -1, 3)
        self.sim.action_tensor().to_torch().copy_(actions)
        self.sim.step()

    def reset(self, env_id, seed = None, options=None):
        self.sim.reset(env_id)
        return

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


class Convolutional1D(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=16, output_size=16,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        # self.num_actions = env.single_action_space.shape[0]
        self.channels_last = channels_last
        self.downsample = downsample
        self.action_space_type = env.action_space_type
        self.num_features = env.num_obs_features


        # self.initial_norm = nn.BatchNorm1d(framestack)
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_features, hidden_size)),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, output_size)),
            nn.ReLU()
        #     nn.ReLU(),
        #     nn.BatchNorm1d(2),  # Normalization layer after the first convolution
        #     pufferlib.pytorch.layer_init(nn.Conv1d(2, 4, 16, stride=1)),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(4),  # Normalization layer after the second convolution
        #     pufferlib.pytorch.layer_init(nn.Conv1d(4, 4, 8, stride=1)),
        #     # nn.ReLU(),
        #     # nn.BatchNorm1d(32),  # Normalization layer after the third convolution
        #     # pufferlib.pytorch.layer_init(nn.Conv1d(32, 64, 64, stride=1)),
        #     # nn.ReLU(),
        #     # nn.BatchNorm1d(64),  # Normalization layer after the fourth convolution
        #     # pufferlib.pytorch.layer_init(nn.Conv1d(64, 64, 32, stride=1)),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(4),  # Normalization layer after the fifth convolution
        #     nn.AdaptiveAvgPool1d(32),
        #     nn.Flatten(),
        #     pufferlib.pytorch.layer_init(nn.Linear(4*32, 64)),
        #     # nn.ReLU(),
        #     # pufferlib.pytorch.layer_init(nn.Linear(16*1024, 4*1024)),
        #     nn.ReLU(),
        #     pufferlib.pytorch.layer_init(nn.Linear(64, hidden_size)),
        #     nn.BatchNorm1d(hidden_size)  # Normalization before the final linear layer
        )
        # Discrete
        if (self.action_space_type == "discrete"):
            self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, env.action_space.n), std=0.01)
        elif (self.action_space_type == "continuous"):
            self.mean = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
            self.log_std = nn.Parameter(torch.full((self.num_actions,), -2.0))

        
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

        # continuous

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        F.normalize(observations, p=2, dim=1)
        # observations = observations.unsqueeze(1)  # This adds a channel dimension, resulting in [batch_size, 1, length]
        # observations = self.initial_norm(observations.float())
        return self.network(observations.float()), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        assert(torch.isnan(env_outputs).any() == False)
        hidden, lookup = self.encode_observations(env_outputs)
        # Discrete
        if (self.action_space_type == "discrete"):
            actions, value = self.decode_actions(hidden, lookup)
            return actions, value
        elif (self.action_space_type == "continuous"):
            actions = self.mean(hidden)
            actions = torch.clamp(actions, min=-1, max=1)
            log_std = torch.clamp(self.log_std, min=-5, max=-2)
            std = torch.exp(log_std)
            value = self.value_fn(hidden)
            return actions, value

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

def make_gpudrive(action_space_type: str = "continuous"):
    return GPUDriveEnv(action_space_type=action_space_type)

if __name__ == "__main__":
    env = GPUDriveEnv()
    # print(env.sim.self_observation_tensor().to_jax())
    # return
    frames = []
    for i in range(91):
        env_obs, rews, dones, truncateds, infos, env_ids, mask = env.recv()
        env.sim.step()

        print(i, dones[0], env_obs[0])
    #     frame = env.render()
    #     frame_np = frame.cpu().numpy()[0][0]
    #     print(frame_np.shape)
    #     frames.append(frame_np.astype('uint8'))

    # with imageio.get_writer('video.mp4', fps=30) as video:
    #     for frame in frames:
    #         video.append_data(frame)
    # print("Video saved to video.mp4")