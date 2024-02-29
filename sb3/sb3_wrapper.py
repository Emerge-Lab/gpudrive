from typing import List, Any

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
import torch

from gpudrive_gym_env import Env


class SB3Wrapper(VecEnv):
    def __init__(self, cfg):
        self._env = Env()
        self.num_envs = self._env.num_sims * self._env.num_agents
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.action = None

    def reset(self, seed=None):
        """Flatten the observations for SB3"""
        return self._env.reset().flatten(end_dim=-2)

    def step_async(self, action):
        self.action = action.clone()

    def step_wait(self):
        obs, rew, done, valids = self._env.step(self.action)
        info = {}
        # TODO(ev) this should actually be coming from the env
        info["valids"] = valids
        info["truncated"] = torch.zeros_like(rew).flatten()
        # info["episode"] = {}
        # info["truncated"] = info["truncated"].flatten()
        return obs.flatten(end_dim=-2), rew.flatten(), done.flatten(), info

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