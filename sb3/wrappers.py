from typing import List, Any
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
import torch
import gymnasium as gym
from abc import ABC, abstractmethod

# GPU Drive Gym Environment
from gpudrive_gym_env import Env


class SB3EnvWrapper(gym.Env):
    """Makes the GPU Drive Gym Environment compatible with Stable Baselines 3."""
    def __init__(self, gpu_drive_env: Env):
        
        self._env = gpu_drive_env
        self.num_envs = self._env.num_sims * self._env.max_cont_agents
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, seed=None):
        """Flatten the observations for SB3"""
        # Only take action from a single agent
        obs = self._env.reset()
        
        # Make sure obs fits with stable baselines, 
        # which expects a tensor of shape (num_envs, num_features)
        return obs.reshape(self.num_envs, -1), {}
    
    def step(self, action):
        """Args:
            action (torch.Tensor): (num_worlds, num_features)
        """
        # GPU Drive expects the action in shape (num_worlds, max_cont_agents):
        action = self.action.reshape(self._env.num_sims, self._env.max_cont_agents)
        
        # Take a step in the environment
        obs, rew, done, info = self._env.step(action)
        
        # Reshape the tensors to fit with stable baselines,
        # Which expects a tensor of shape (num_envs, num_features)
        obs = obs.reshape(self.num_envs, -1)
        rew = rew.reshape(self.num_envs, -1)
        done = done.reshape(self.num_envs, -1)
        
        return obs, rew, done, info

    @property
    def action_space(self):
        return self._env.action_space

    @action_space.setter
    def action_space(self, action_space):
        self._env.action_space = action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._env.observation_space = observation_space

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_attr(self, attr_name: str):
        return getattr(self._env, attr_name)

    def set_attr(self, attr_name: str):
        setattr(self._env, attr_name)

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> List[bool]:
        return super().env_is_wrapped(wrapper_class, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
    
    
if __name__ == "__main__":
    
    env = Env(num_worlds=1, max_cont_agents=1, device='cuda')
    env = SB3EnvWrapper(env)
    
    obs = env.reset()
    
    action = torch.ones((env.num_envs))
    
    obs, rew, done, info = env.step(action)
    
    
          