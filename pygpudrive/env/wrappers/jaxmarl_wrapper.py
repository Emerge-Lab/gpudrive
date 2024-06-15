"""
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

import jax
import jax.numpy as jnp
from typing import Dict
import chex
from functools import partial
from flax import struct
from typing import Tuple, Optional

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.env_jax import BaseEnvJax


@struct.dataclass
class State:
    done: chex.Array
    step: int


class GPUDriveToJaxMARL(object):
    """
    Wrapper to make the GPUDrive base environment class compatible with JaxMARL.
    """

    def __init__(
        self,
        env: BaseEnvJax,
    ) -> None:
        self.env = env
        self.num_agents = env.config.max_agent_count
        self.observation_spaces = env.observation_space
        self.action_spaces = ...

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict
    ]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(
            key, state, actions
        )

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            states_re,
            states_st,
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[
        Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict
    ]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError


if __name__ == "__main__":

    env_config = EnvConfig()

    base_env = BaseEnvJax(
        config=env_config,
        num_worlds=10,
        max_cont_agents=128,
        data_dir="example_data",
    )

    jaxmarl_env = GPUDriveToJaxMARL(base_env)
