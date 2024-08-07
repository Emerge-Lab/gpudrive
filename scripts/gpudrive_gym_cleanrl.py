import gpudrive
import gymnasium as gym

from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn
import numpy as np
from gymnasium.experimental.wrappers import LambdaActionV0

from sim_utils.creator import SimCreator
from itertools import product

from box import Box

class GPUDriveEnv(gym.Env):
    Recurrent = None

    def __init__(self, config: dict[str,dict[str,str]] = {}):
        if len(config) > 0:
            self.sim, self.config = SimCreator(config)
        else:
            self.sim, self.config = SimCreator()
            self.config = Box(self.config)
            print(self.config)
        
        self.device = str.lower(self.config.sim_manager.exec_mode)
        
        self.setup_obs()
        self.setup_actions()

        self.async_reset()
        
        self.controlled_num_agents = torch.sum(self.mask).item()
        self.num_obs_features = self.concatenated_obs.shape[-1]
        self.single_observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.num_obs_features], dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.batch_size)

        self.action_space_type = self.config.env_params.action_space_type
        self.single_action_space = gym.spaces.Box(low=-5, high=5, shape=self.batch_action_tensor.shape[1:], dtype=np.float32)
        self.batched_space = gym.vector.utils.batch_space(self.single_action_space, self.batch_size)
        self.action_space = self.batched_space
        if self.action_space_type == "discrete":
            self.setup_discrete_actions()

        # self.agents_per_env = self.shape_tensor[:,0]
        # self.total_num_agents = torch.sum(self.agents_per_env)
        # self.env_ids = torch.repeat_interleave(torch.arange(self.num_envs).to(self.self_obs_tensor.device), self.agents_per_env)

        # self.single_observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.num_obs_features], dtype=np.float64)
        # # self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.num_obs_features,), dtype=torch.float32)
        # self.observation_space = gym.spaces.Dict({
        #     "self_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[0].shape, dtype=np.float64),
        #     "partner_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[1].shape, dtype=np.float64),
        #     "map_obs": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[2].shape, dtype=np.float64),
        #     "steps_remaining": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[3].shape, dtype=np.float64),
        #     "id": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=self.obs_tensors[4].shape, dtype=np.float64),
        # })



        # self.Recurrent = None
        # self.unflatten_context = None
        # self.mask_agents = True

    def setup_discrete_actions(self):
        """Configure the discrete action space."""

        self.steer_actions = torch.tensor([-0.6, 0, 0.6], device=self.self_obs_tensor.device)
        self.accel_actions = torch.tensor([-5 ,-3, -1, 0.5, 0.1, 0, 0.1, 0.5, 1, 3, 5], device=self.self_obs_tensor.device)
        self.head_actions = torch.tensor([0], device = self.self_obs_tensor.device)

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}

        for action_idx, (accel, steer, head) in enumerate(
            product(self.accel_actions, self.steer_actions, self.head_actions)
        ):
            self.action_key_to_values[action_idx] = torch.tensor([
                accel.item(),
                steer.item(),
                head.item(),
            ], device=self.self_obs_tensor.device)

        self.discrete_action_space = Discrete(len(self.action_key_to_values))
        return

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
    
    def render(self):
        rgb_image = self.sim.rgb_tensor().to_torch()
        return rgb_image

    def render_depth(self):
        depth_image = self.sim.depth_tensor().to_torch()
        return depth_image

    def setup_obs(self):
        self.self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        N, A, O = self.self_obs_tensor.shape[0:3] # N = num worlds, A = num agents, O = num obs features
        self.num_envs = N
        self.num_agents = A
        self.batch_size = N * A
        self.obs_type = 'lidar' if self.config['parameters']['enableLidar'] else 'classic'
        if self.obs_type == 'lidar':
            self.lidar_tensor = self.sim.lidar_tensor().to_torch()


            self.obs_tensors = [
                self.self_obs_tensor.view(self.batch_size, *self.self_obs_tensor.shape[2:]),
                self.lidar_tensor.view(self.batch_size, -1),
            ]
        else:
            self.partner_obs_tensor = self.sim.partner_observations_tensor().to_torch()
            self.agent_map_obs_tensor = self.sim.agent_roadmap_tensor().to_torch()
           
            map_obs_tensor = self.agent_map_obs_tensor.view(N, A, -1)
            self.partner_obs_tensor = self.partner_obs_tensor.view(N, A, -1)

            self.obs_tensors = [
                self.self_obs_tensor.view(self.batch_size, *self.self_obs_tensor.shape[2:]),
                self.partner_obs_tensor.view(self.batch_size, *self.partner_obs_tensor.shape[2:]),
                map_obs_tensor.view(self.batch_size, *map_obs_tensor.shape[2:])
            ]

        self.controlled_state_tensor = self.sim.controlled_state_tensor().to_torch()
        self.done_tensor = self.sim.done_tensor().to_torch()
        self.reward_tensor = self.sim.reward_tensor().to_torch()
        self.steps_remaining_tensor = self.sim.steps_remaining_tensor().to_torch()

        # Add in an agent ID tensor
        self.id_tensor = torch.arange(N*A, dtype=torch.int64)
        self.id_tensor = self.id_tensor.to(device=self.self_obs_tensor.device)
        self.id_tensor = self.id_tensor.view(N, A).expand(N, A).reshape(self.batch_size, 1)

        
        self.controlled = self.controlled_state_tensor.view(self.batch_size, *self.controlled_state_tensor.shape[2:])
        self.done = self.done_tensor.view(self.batch_size, *self.done_tensor.shape[2:])
        self.reward = self.reward_tensor.view(self.batch_size, *self.reward_tensor.shape[2:])
        self.steps_remaining = self.steps_remaining_tensor.view(self.batch_size, *self.steps_remaining_tensor.shape[2:])
        self.agent_ids = self.id_tensor
        self.info_tensor = self.sim.info_tensor().to_torch()
        self.info = self.info_tensor.view(self.batch_size, *self.info_tensor.shape[2:])

        controlled_mask = self.controlled == 1
        done_mask = self.done != 1
        self.mask = torch.clone(torch.squeeze(controlled_mask & done_mask)).detach()
        self.prev_mask = self.mask

    @property
    def concatenated_obs(self):
        # TODO: Memory copy here. Can we avoid it?
        return torch.cat(self.obs_tensors, dim=-1)

    def setup_actions(self):
        self.action_tensor = self.sim.action_tensor().to_torch()
        self.batch_action_tensor = self.action_tensor.view(self.batch_size, *self.action_tensor.shape[2:])
        return

    def step(self, actions):
        acs = torch.zeros_like(self.batch_action_tensor)
        acs[self.prev_mask] = actions
        acs = acs.view(self.num_envs, self.num_agents, *acs.shape[1:])
        self.send(acs)
        return
        # action = torch.tensor(action)
        # self.sim.step()
        # obs, _ = self.setup_obs()
        # obs = {
        #     "self_obs": obs[0].numpy(),
        #     "partner_obs": obs[1].numpy(),
        #     "map_obs": obs[2].numpy(),
        #     "steps_remaining": obs[3].numpy(),
        #     "id": obs[4].numpy(),
        # }
        # reward = torch.sum(self.sim.reward_tensor().to_torch()).numpy().astype(np.float32).item()
        # done = self.sim.done_tensor().to_torch()
        # info = {}
        # return obs, reward, done.any().item(), False, info
    
    def async_reset(self, seed = None):
        # seed is ignored. The sim is deterministic.
        # self.data = [self.reset(i) for i in range(self.num_envs)]
        for i in range(self.num_envs):
            self.sim.reset(i)
        self.sim.step()
        self.make_mask()
        return self.recv()
    
    def make_mask(self):
        controlled_mask = self.controlled == 1
        done_mask = self.done != 1
        self.prev_mask = torch.clone(self.mask)
        self.mask = torch.clone(torch.squeeze(controlled_mask & done_mask)).detach()

    def get_info(self):
        goal_reach = torch.sum(self.info[self.prev_mask, 3]).item()
        collisions = torch.sum(self.info[self.prev_mask, :3]).item()
        return {"goal_reach": goal_reach, "collisions": collisions}

    @torch.no_grad()
    def recv(self):
        self.make_mask()
        # self.reward[self.prev_mask] = self.reward[self.prev_mask] - torch.sum(self.info[self.prev_mask,:3],dim = 1,keepdim=True)
        return self.concatenated_obs[self.prev_mask], self.reward[self.prev_mask], self.done[self.prev_mask], self.done[self.prev_mask], [self.get_info()], self.agent_ids[self.prev_mask], self.prev_mask

    def send(self, actions):
        self.sim.action_tensor().to_torch().copy_(actions)
        self.sim.step()

    def reset(self, seed = None, options=None):
        self.async_reset()
        return self.recv()

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
    
    def close(self):
        del self.sim
        return super().close()


def apply_discrete_action(actions, action_key_to_values):
    actions = torch.as_tensor(actions)
    device = actions.device
    if(isinstance(actions, np.ndarray)):
        actions = torch.from_numpy(actions)
    action_value_tensor = torch.zeros(actions.shape[0], 3, device=device)
    for idx, action in enumerate(actions):
        action_idx = action.item()
        action_value_tensor[idx, :] = action_key_to_values[action_idx]
    return action_value_tensor

def make_gpudrive(config: dict[str,dict[str,str]] = {}):
    env = GPUDriveEnv(config)
    if(env.action_space_type == "discrete"):
        env = LambdaActionV0(
            env,
            func=lambda action: apply_discrete_action(action, env.unwrapped.action_key_to_values),
            action_space=gym.vector.utils.batch_space(env.discrete_action_space, env.batch_size)
        )
    return env

if __name__ == "__main__":
    env = make_gpudrive()
    print(env.action_space_type)
    print(env.discrete_action_space.n)
    # env_obs, rews, dones, truncateds, infos, agent_ids, mask = env.recv()
    # # print(env.sim.self_observation_tensor().to_jax())
    # # return
    # frames = []
    # for i in range(91):
    #     actions = torch.tensor(env.action_space.sample())
    #     env.step(actions[mask])
    #     env_obs, rews, dones, truncateds, infos, agent_ids, mask = env.recv()
    #     if(dones.all()):
    #         break
    #     print(env_obs[0][0], env.sim.self_observation_tensor().to_torch()[0][0])

        # print(i, dones[0], env_obs[0])
    #     frame = env.render()
    #     frame_np = frame.cpu().numpy()[0][0]
    #     print(frame_np.shape)
    #     frames.append(frame_np.astype('uint8'))

    # with imageio.get_writer('video.mp4', fps=30) as video:
    #     for frame in frames:
    #         video.append_data(frame)
    # print("Video saved to video.mp4")