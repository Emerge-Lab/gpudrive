import os
from typing import List, Optional
import gymnasium as gym
from pygpudrive.env.config import RenderConfig, RenderMode
from pygpudrive.env.viz import PyGameVisualizer
from pygpudrive.env.scene_selector import select_scenes
import abc
import gpudrive
import torch
import jax.numpy as jnp


class GPUDriveGymEnv(gym.Env, metaclass=abc.ABCMeta):
    def __init__(self, backend="torch"):
        super().__init__()
        self.backend = backend
        if self.backend not in ["torch", "jax"]:
            raise ValueError("Unsupported backend; use 'torch' or 'jax'")

    def to_tensor(self, x):
        """Convert simulator data to the correct tensor type for the specified backend."""
        if self.backend == "torch":
            return x.to_torch()
        elif self.backend == "jax":
            return x.to_jax()

    @abc.abstractmethod
    def reset(self):
        """Reset the dynamics to inital state.
        Args:
            scenario: Scenario used to generate the initial state.
            rng: Optional random number generator for stochastic environments.

            Returns:
                The initial observations.
        """

    @abc.abstractmethod
    def step_dynamics(self, actions):
        """Advance the dynamics by one step.

        Args:
            actions: The actions to apply to the environment.

        Returns: None.
        """

    @abc.abstractmethod
    def get_dones():
        """Returns the done flags for the environment."""

    @abc.abstractmethod
    def get_infos():
        """Returns the info tensor for the environment."""

    @abc.abstractmethod
    def get_rewards():
        """Returns the reward tensor for the environment."""

    def _set_reward_params(self):
        """Configures the reward parameters based on environment settings.

        Returns:
            object: Configured reward parameters.
        """
        reward_params = gpudrive.RewardParams()

        if (
            self.config.reward_type == "sparse_on_goal_achieved"
            or self.config.reward_type == "weighted_combination"
        ):
            reward_params.rewardType = gpudrive.RewardType.OnGoalAchieved
        else:
            raise ValueError(f"Invalid reward type: {self.config.reward_type}")

        reward_params.distanceToGoalThreshold = (
            self.config.dist_to_goal_threshold
        )
        return reward_params

    def _set_road_reduction_params(self, params):
        """Configures the road reduction parameters.

        Args:
            params (object): Parameters object to be modified.

        Returns:
            object: Updated parameters object with road reduction settings.
        """
        params.observationRadius = self.config.obs_radius
        if self.config.road_obs_algorithm == "k_nearest_roadpoints":
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.KNearestEntitiesWithRadiusFiltering
            )
        else:  # Default to linear algorithm
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering
            )
        return params

    def _setup_environment_parameters(self):
        """Sets up various parameters required for the environment simulation.

        Returns:
            object: Configured parameters for the simulation.
        """
        # Dict with supported dynamics models
        self.dynamics_model_dict = dict(
            classic=gpudrive.DynamicsModel.Classic,
            delta_local=gpudrive.DynamicsModel.DeltaLocal,
            bicycle=gpudrive.DynamicsModel.InvertibleBicycle,
            state=gpudrive.DynamicsModel.State,
        )

        params = gpudrive.Parameters()
        params.polylineReductionThreshold = (
            self.config.polyline_reduction_threshold
        )
        params.rewardParams = self._set_reward_params()
        params.maxNumControlledAgents = self.max_cont_agents
        if self.config.init_all_objects:
            params.isStaticAgentControlled = True
            params.initOnlyValidAgentsAtFirstStep = False
            params.IgnoreNonVehicles = False
        else:
            params.isStaticAgentControlled = False
            params.initOnlyValidAgentsAtFirstStep = True
            params.IgnoreNonVehicles = self.config.remove_non_vehicles
        params.dynamicsModel = self.dynamics_model_dict[
            self.config.dynamics_model
        ]
        if self.config.dynamics_model not in self.dynamics_model_dict:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

        if self.config.lidar_obs:
            if not self.config.lidar_obs and self.config.disable_classic_obs:
                raise ValueError(
                    "Lidar observations must be enabled if classic observations are disabled."
                )

            else:
                params.enableLidar = self.config.lidar_obs
                params.disableClassicalObs = self.config.disable_classic_obs
                self.config.ego_state = False
                self.config.road_map_obs = False
                self.config.partner_obs = False
        params = self._set_collision_behavior(params)
        params = self._set_road_reduction_params(params)

        # Map entity types to integers
        self.ENTITY_TYPE_TO_INT = {
            gpudrive.EntityType._None: 0,
            gpudrive.EntityType.RoadEdge: 1,
            gpudrive.EntityType.RoadLine: 2,
            gpudrive.EntityType.RoadLane: 3,
            gpudrive.EntityType.CrossWalk: 4,
            gpudrive.EntityType.SpeedBump: 5,
            gpudrive.EntityType.StopSign: 6,
            gpudrive.EntityType.Vehicle: 7,
            gpudrive.EntityType.Pedestrian: 8,
            gpudrive.EntityType.Cyclist: 9,
            gpudrive.EntityType.Padding: 10,
        }
        self.MIN_OBJ_ENTITY_ENUM = min(list(self.ENTITY_TYPE_TO_INT.values()))
        self.MAX_OBJ_ENTITY_ENUM = max(list(self.ENTITY_TYPE_TO_INT.values()))
        self.ROAD_MAP_OBJECT_TYPES = 7  # (enums 0-6)
        self.ROAD_OBJECT_TYPES = 4  # (enums 7-10)

        return params

    def _initialize_simulator(self, params, scene_config):
        """Initializes the simulation with the specified parameters.

        Args:
            params (object): Parameters for initializing the simulation.

        Returns:
            SimManager: A simulation manager instance configured with given parameters.
        """
        exec_mode = (
            gpudrive.madrona.ExecMode.CPU
            if self.device == "cpu"
            else gpudrive.madrona.ExecMode.CUDA
        )

        self.dataset = select_scenes(scene_config)
        sim = gpudrive.SimManager(
            exec_mode=exec_mode,
            gpu_id=0,
            scenes=self.dataset,
            params=params,
            enable_batch_renderer=self.render_config
            and self.render_config.render_mode
            in {RenderMode.MADRONA_RGB, RenderMode.MADRONA_DEPTH},
            batch_render_view_width=self.render_config.resolution[0]
            if self.render_config
            else None,
            batch_render_view_height=self.render_config.resolution[1]
            if self.render_config
            else None,
        )

        return sim

    def _setup_rendering(self):
        """Sets up the rendering mechanism based on the configuration.

        Returns:
            PyGameVisualizer: A visualizer instance for rendering the environment.
        """
        return PyGameVisualizer(
            self.sim, self.render_config, self.config.dist_to_goal_threshold
        )

    def _setup_action_space(self, action_type):
        """Sets up the action space based on the specified type.

        Args:
            action_type (str): Type of action space to set up.

        Raises:
            ValueError: If the specified action type is not supported.
        """
        if action_type == "discrete":
            self.action_space = self._set_discrete_action_space()
        elif action_type == "continuous":
            self.action_space = self._set_continuous_action_space()
        else:
            raise ValueError(f"Action space not supported: {action_type}")

    def _set_collision_behavior(self, params):
        """Defines the behavior when a collision occurs.

        Args:
            params (object): Parameters object to update based on collision behavior.

        Returns:
            object: Updated parameters with collision behavior settings.
        """
        if self.config.collision_behavior == "ignore":
            params.collisionBehaviour = gpudrive.CollisionBehaviour.Ignore
        elif self.config.collision_behavior == "remove":
            params.collisionBehaviour = (
                gpudrive.CollisionBehaviour.AgentRemoved
            )
        elif self.config.collision_behavior == "stop":
            params.collisionBehaviour = gpudrive.CollisionBehaviour.AgentStop
        else:
            raise ValueError(
                f"Invalid collision behavior: {self.config.collision_behavior}"
            )
        return params

    def render(self, world_render_idx=0, color_objects_by_actor=None):
        """Renders the environment.

        Args:
            world_render_idx (int): Index of the world to render.

        Returns:
            Any: Rendered view of the world, or None if an invalid index is specified.
        """
        if world_render_idx >= self.num_worlds:
            print(f"Invalid world_render_idx: {world_render_idx}")
            return None
        if self.render_config.render_mode in {
            RenderMode.PYGAME_ABSOLUTE,
            RenderMode.PYGAME_EGOCENTRIC,
            RenderMode.PYGAME_LIDAR,
        }:
            return self.visualizer.getRender(
                world_render_idx=world_render_idx,
                cont_agent_mask=self.cont_agent_mask,
                color_objects_by_actor=color_objects_by_actor,
            )
        elif self.render_config.render_mode in {
            RenderMode.MADRONA_RGB,
            RenderMode.MADRONA_DEPTH,
        }:
            return self.visualizer.getRender()

    def reinit_scenarios(self, dataset: List[str]):
        """Resample the scenes.
        Args:
            dataset (List[str]): List of scene names to resample.

        Returns:
            None
        """

        # Resample the scenes
        self.sim.set_maps(dataset)

        # Re-initialize the controlled agents mask
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

    def close(self):
        """Destroy the simulator and visualizer."""
        del self.sim

    def normalize_tensor(self, x, min_val, max_val):
        """Normalizes an array of values to the range [-1, 1].

        Args:
            x (np.array): Array of values to normalize.
            min_val (float): Minimum value for normalization.
            max_val (float): Maximum value for normalization.

        Returns:
            np.array: Normalized array of values.
        """
        return 2 * ((x - min_val) / (max_val - min_val)) - 1
