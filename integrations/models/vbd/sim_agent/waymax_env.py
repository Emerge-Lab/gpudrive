# This class is a wrapper of Waymax to simulate the environment from WOMD

import numpy as np
from jax import jit
from jax import numpy as jnp

from waymax import config as _config
from waymax import datatypes
from waymax import dynamics
from waymax import env as waymax_env
from waymax import agents
from waymax.agents import actor_core

from typing import List


class WaymaxEnvironment(waymax_env.BaseEnvironment):
    def __init__(
        self,
        dynamics_model: dynamics.DynamicsModel,
        config: _config.EnvironmentConfig,
        log_replay=True,
    ):
        """
        Initializes a new instance of the WaymaxEnv class.

        Args:
            dynamics_model (dynamics.DynamicsModel): The dynamics model used for simulating the environment.
            config (_config.EnvironmentConfig): The configuration object for the environment.
        """
        super().__init__(dynamics_model, config)

        # override the reward function with the dictionary reward function
        self._dynamics_model = dynamics_model
        self.config = config
        if log_replay:
            self.nc_actor = agents.create_expert_actor(
                dynamics_model=dynamics_model,
                is_controlled_func=lambda state: state.object_metadata.is_valid,
            )
        else:
            # create a CS actor for agent. Sim Agent will be filtered out during step as a post-processing step
            self.nc_actor = agents.create_expert_actor(
                dynamics_model=dynamics_model,
                is_controlled_func=lambda state: state.object_metadata.is_valid,
            )

        # Useful jited functions
        self.jit_step = jit(self.step)
        self.jit_nc_action = jit(self.nc_actor.select_action)
        self.jit_reset = jit(super().reset)

    def step_sim_agent(
        self,
        current_state: datatypes.SimulatorState,
        sim_agent_action_list: List[datatypes.Action],
    ) -> datatypes.SimulatorState:
        """
        Steps the simulation agent.
        """
        # Step the CS Policy
        nc_action_full: datatypes.Action = self.jit_nc_action(
            {}, current_state, None, None
        )

        # do a validation check
        is_controlled_stack = jnp.vstack(
            [action.is_controlled for action in sim_agent_action_list]
        )
        num_controlled = jnp.sum(is_controlled_stack, axis=0)  # (num_agent, 1)
        if jnp.any(num_controlled > 1):
            raise Exception("An agent is controlled by more than one policy")

        # set the is_controlled flag
        simple_control = num_controlled == 0

        nc_action = actor_core.WaymaxActorOutput(
            action=nc_action_full.action,
            actor_state=None,
            is_controlled=simple_control,
        )

        # merge the actions
        sim_agent_action_list.append(nc_action)
        action_merged = agents.merge_actions(sim_agent_action_list)

        # step the environment
        next_state = self.jit_step(current_state, action_merged)
        next_state.object_metadata.is_controlled = num_controlled > 0

        return next_state

    def reset(
        self, state: datatypes.SimulatorState
    ) -> datatypes.SimulatorState:
        """Initializes the simulation state.

        This initializer sets the initial timestep and fills the initial simulation
        trajectory with invalid values.

        Args:
        state: An uninitialized state of shape (...).

        Returns:
        The initialized simulation state of shape (...).
        """
        return self.jit_reset(state)
