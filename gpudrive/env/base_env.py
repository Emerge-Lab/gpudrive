import gymnasium as gym
from gpudrive.env.config import RenderMode
import madrona_gpudrive
import abc


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
        reward_params = madrona_gpudrive.RewardParams()

        if (
            self.config.reward_type == "sparse_on_goal_achieved"
            or self.config.reward_type == "weighted_combination"
            or self.config.reward_type == "guided_autonomy"
            or self.config.reward_type == "reward_conditioned"
        ):
            reward_params.rewardType = (
                madrona_gpudrive.RewardType.OnGoalAchieved
            )
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
        params.viewConeHalfAngle = self.config.view_cone_half_angle
        params.viewOccludeObjects = self.config.view_occlude_objects
        if self.config.road_obs_algorithm == "k_nearest_roadpoints":
            params.roadObservationAlgorithm = (
                madrona_gpudrive.FindRoadObservationsWith.KNearestEntitiesWithRadiusFiltering
            )
        else:  # Default to linear algorithm
            params.roadObservationAlgorithm = (
                madrona_gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering
            )
        return params

    def _setup_environment_parameters(self):
        """Sets up various parameters required for the environment simulation.

        Returns:
            object: Configured parameters for the simulation.
        """
        # Dict with supported dynamics models
        self.dynamics_model_dict = dict(
            classic=madrona_gpudrive.DynamicsModel.Classic,
            delta_local=madrona_gpudrive.DynamicsModel.DeltaLocal,
            bicycle=madrona_gpudrive.DynamicsModel.InvertibleBicycle,
            state=madrona_gpudrive.DynamicsModel.State,
        )

        params = madrona_gpudrive.Parameters()

        params.polylineReductionThreshold = (
            self.config.polyline_reduction_threshold
        )
        params.rewardParams = self._set_reward_params()
        params.maxNumControlledAgents = self.max_cont_agents
        if self.config.init_mode == "wosac_eval":
            # Bypasses all gpudrive initialization rules and directly reads from the tracks_to_predict
            # flag in the WOMD dataset metadata
            params.readFromTracksToPredict = True
            params.isStaticAgentControlled = True
            params.controlExperts = True
        elif self.config.init_mode == "wosac_train":
            params.readFromTracksToPredict = True
            params.isStaticAgentControlled = True
            params.controlExperts = False
        elif self.config.init_mode == "all_objects":
            params.isStaticAgentControlled = True
            params.initOnlyValidAgentsAtFirstStep = False
            params.IgnoreNonVehicles = False
        elif self.config.init_mode == "all_valid":
            params.isStaticAgentControlled = True
            params.initOnlyValidAgentsAtFirstStep = True
            params.IgnoreNonVehicles = self.config.remove_non_vehicles
        elif self.config.init_mode == "all_non_trivial":
            params.isStaticAgentControlled = False
            params.initOnlyValidAgentsAtFirstStep = True
            params.IgnoreNonVehicles = self.config.remove_non_vehicles
        else:
            raise ValueError(f"Invalid init mode: {self.config.init_mode}")

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
        params = self._set_goal_behavior(params)
    
        if self.config.guidance and self.config.guidance_mode == "vbd_online":
            self.init_steps = max(self.config.init_steps, 10)
            print(
                f"\n[Note] Guidance mode '{self.config.guidance_mode}' requires at least 10 initialization steps to provide sufficient scene context for the diffusion model. Automatically setting simulator time to t = {self.init_steps}. \n"
            )
        else:
            self.init_steps = getattr(self.config, "init_steps", 0)
            
        params.initSteps = self.init_steps

        return params

    def _initialize_simulator(self, params, data_batch):
        """Initializes the simulation with the specified parameters.

        Args:
            params (object): Parameters for initializing the simulation.

        Returns:
            SimManager: A simulation manager instance configured with given parameters.
        """
        exec_mode = (
            madrona_gpudrive.madrona.ExecMode.CPU
            if self.device == "cpu"
            else madrona_gpudrive.madrona.ExecMode.CUDA
        )

        sim = madrona_gpudrive.SimManager(
            exec_mode=exec_mode,
            gpu_id=0,
            scenes=data_batch,
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
            params.collisionBehaviour = (
                madrona_gpudrive.CollisionBehaviour.Ignore
            )
        elif self.config.collision_behavior == "remove":
            params.collisionBehaviour = (
                madrona_gpudrive.CollisionBehaviour.AgentRemoved
            )
        elif self.config.collision_behavior == "stop":
            params.collisionBehaviour = (
                madrona_gpudrive.CollisionBehaviour.AgentStop
            )
        else:
            raise ValueError(
                f"Invalid collision behavior: {self.config.collision_behavior}"
            )
        return params

    def _set_goal_behavior(self, params):
        """Configures the behavior when an agent reaches its goal.

        Args:
            params (object): Parameters object to update based on goal behavior.

        Returns:
            object: Updated parameters with goal behavior settings.
        """
        if self.config.goal_behavior == "remove":
            params.goalBehaviour = madrona_gpudrive.GoalBehaviour.Remove
        elif self.config.goal_behavior == "stop":
            params.goalBehaviour = madrona_gpudrive.GoalBehaviour.Stop
        elif self.config.goal_behavior == "ignore":
            params.goalBehaviour = madrona_gpudrive.GoalBehaviour.Ignore
        else:
            raise ValueError(
                f"Invalid goal behavior: {self.config.goal_behavior}"
            )
        return params

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
