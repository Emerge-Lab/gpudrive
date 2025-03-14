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

from gpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from gpudrive.env.env_jax import GPUDriveJaxEnv


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
        env: GPUDriveJaxEnv,
    ) -> None:
        self.env = env
        self.max_agents = (
            env.max_agent_count
        )  # Max number of POSSIBLE agents in the environment
        self.controlled_agent_mask = (
            env.cont_agent_mask
        )  # Agents across all worlds that are controlled
        self.episode_len = 90  # Maximum number of steps in an episode
        self.controlled_agents_across_worlds = (
            env.cont_agent_mask.sum().item()
        )  # All controlled agents
        self.observation_spaces = {
            i: env.observation_space
            for i in range(self.controlled_agents_across_worlds)
        }
        self.action_spaces = {
            i: env.action_space
            for i in range(self.controlled_agents_across_worlds)
        }
        # Note (dc): The ordering of the agents will remain the same, even when agents die
        self.agents = [
            f"agent_{i}" for i in range(self.controlled_agents_across_worlds)
        ]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        # Note (dc): There is currently no randomness when we reset, so we don't need a key
        obs = self.env.reset()

        masked_dict_obs = self.get_obs(obs)

        # Initialize the dead agent mask
        self.dead_agent_mask = self.controlled_agent_mask.copy()

        # obs is of shape (num_agents, obs_dim)
        # Note (dc): we're not returning the whole simulator state, only the observation
        # for every agent in the environment
        return masked_dict_obs, None

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: None,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict
    ]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, _, rewards, dones, infos = self.step_env(key, state, actions)
        obs_re, _ = self.reset(key_reset)

        states = None

        # Auto-reset environment based on termination
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

        # Note (dc): Convert input action dictionary to array of actions
        action_arr = jnp.stack(
            [actions[agent_name] for agent_name in self.agents]
        ).reshape(self.controlled_agents_across_worlds)

        # Step the environment dynamics
        # Note (dc): We could filter out the actions for invalid agents for correctness,
        # however, the simulator internally ignores actions for dead or invalid agents
        # so this is not strictly necessary
        self.env.step_dynamics(action_arr)

        # Get the observations for all agents
        obs = self.get_obs(self.env.get_obs())
        reward = self.env.get_rewards()
        done = self.env.get_dones()
        info = self.env.get_infos()

        # Mask the observations, rewards, dones, and infos for agents and map to dicts
        # that are not controlled (padding agents) OR are done
        rewards_masked = reward[self.controlled_agent_mask]
        rewards = {
            agent: rewards_masked[agent_idx]
            for agent_idx, agent in enumerate(self.agents)
        }
        rewards["__all__"] = jnp.sum(rewards_masked)

        # Note: done is 1 from the moment the agent is done and all subsequent time steps (before we reset)
        # The maximum episode length is 90 steps
        dones_masked = done[self.controlled_agent_mask]
        dones = {
            agent: dones_masked[agent_idx]
            for agent_idx, agent in enumerate(self.agents)
        }
        dones["__all__"] = jnp.all(dones_masked)

        infos_masked = info[self.controlled_agent_mask]
        infos = {
            agent: infos_masked[agent_idx]
            for agent_idx, agent in enumerate(self.agents)
        }

        # Update the dead agent mask
        # self.dead_agent_mask = jnp.logical_or(self.dead_agent_mask, done)

        return obs, None, rewards, dones, infos

    def get_obs(self, state) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        # Flatten along agent x world dimensions
        valid_obs = state[self.controlled_agent_mask].reshape(
            self.controlled_agents_across_worlds, -1
        )

        agent_obs = {}
        for agent_idx, agent_name in enumerate(self.agents):
            agent_obs[agent_name] = valid_obs[agent_idx, :]

        return agent_obs

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
    render_config = RenderConfig()
    scene_config = SceneConfig(path="data", num_scenes=3)

    # MAKE ENV
    base_env = GPUDriveJaxEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=128,  # Number of agents to control
        device="cuda",
        render_config=render_config,
    )

    jaxmarl_env = GPUDriveToJaxMARL(base_env)

    # RESET
    jaxmarl_env.reset(key=jax.random.PRNGKey(0))

    for step in range(10):
        print(f"step: {step}")

        # STEP
        jaxmarl_env.step(
            key=jax.random.PRNGKey(0),
            state=None,
            actions={agent: jnp.zeros(1) for agent in jaxmarl_env.agents},
        )
