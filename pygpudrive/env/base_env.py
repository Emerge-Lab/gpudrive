import os
import gymnasium as gym
from pygpudrive.env.config import RenderConfig, RenderMode
from pygpudrive.env.viz import PyGameVisualizer
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

    def _validate_data_dir(self):
        """Validates that the data directory exists and is not empty.

        Raises:
            AssertionError: If the data directory does not exist or is empty.
        """
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            assert False, "The data directory does not exist or is empty."

    def _setup_environment_parameters(self):
        """Sets up various parameters required for the environment simulation.

        Returns:
            object: Configured parameters for the simulation.
        """
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = (
            self.config.polyline_reduction_threshold
        )
        params.rewardParams = self._set_reward_params()
        params.IgnoreNonVehicles = self.config.remove_non_vehicles
        params.maxNumControlledVehicles = self.max_cont_agents
        params.isStaticAgentControlled = False
        params.useWayMaxModel = False

        if self.config.enable_lidar:
            params.enableLidar = self.config.enable_lidar
            params.disableClassicalObs = True

        params = self._set_collision_behavior(params)
        params = self._init_dataset(params, self.data_dir)
        params = self._set_road_reduction_params(params)
        return params

    def _initialize_simulator(self, params):
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
        return gpudrive.SimManager(
            exec_mode=exec_mode,
            gpu_id=0,
            num_worlds=self.num_worlds,
            json_path=self.data_dir,
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

    def _init_dataset(self, params, data_dir):
        """Initializes the dataset based on sampling method specified in the configuration.

        Args:
            params (object): Parameters object to update with dataset initialization options.
            data_dir (str): Path to the directory containing the dataset.

        Returns:
            object: Updated parameters with dataset initialization options.
        """
        if self.config.sample_method == "first_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.FirstN
        elif self.config.sample_method == "random_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.RandomN
        elif self.config.sample_method == "pad_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.PadN
        elif self.config.sample_method == "exact_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.ExactN

        self.data_dir = data_dir

        return params

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

    def render(self, world_render_idx=0):
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
            )
        elif self.render_config.render_mode in {
            RenderMode.MADRONA_RGB,
            RenderMode.MADRONA_DEPTH,
        }:
            return self.visualizer.getRender()

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
