import os
import gymnasium as gym
from pygpudrive.env.config import RenderConfig, RenderMode
from pygpudrive.env.viz import PyGameVisualizer
from pygpudrive.env.scene_selector import select_scenes
import abc
import gpudrive

class GPUDriveGymEnv(gym.Env, metaclass=abc.ABCMeta):
    """Base class for multi-agent environments in GPUDrive.

    Provides common methods for setting up the simulator and handling output.
    """

    def __init__(self):
        super().__init__()

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

        if self.config.reward_type == "sparse_on_goal_achieved":
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
        self.dynamics_model = dict(
            classic = gpudrive.DynamicsModel.Classic,
            delta = gpudrive.DynamicsModel.Delta,
            waymax = gpudrive.DynamicsModel.Waymax,
        )

        params = gpudrive.Parameters()
        params.polylineReductionThreshold = (
            self.config.polyline_reduction_threshold
        )
        params.rewardParams = self._set_reward_params()
        params.IgnoreNonVehicles = self.config.remove_non_vehicles
        params.maxNumControlledVehicles = self.max_cont_agents
        params.isStaticAgentControlled = False
        params.dynamicsModel = self.dynamics_model[self.config.dynamics_model]

        if self.config.enable_lidar:
            params.enableLidar = self.config.enable_lidar
            params.disableClassicalObs = True

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
        self.ROAD_MAP_OBJECT_TYPES = 7 # (enums 0-6)
        self.ROAD_OBJECT_TYPES = 4 # (enums 7-10)

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

        dataset = select_scenes(scene_config)
        sim = gpudrive.SimManager(
            exec_mode=exec_mode,
            gpu_id=0,
            scenes=dataset,
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

    def close(self):
        """Destroy the simulator and visualizer."""
        del self.sim
        self.visualizer.destroy()

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
