"""Torch Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
from itertools import product
import mediapy as media
import gymnasium

from gpudrive.env import constants
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.base_env import GPUDriveGymEnv
from gpudrive.datatypes.trajectory import (
    LogTrajectory,
    VBDTrajectory,
    VBDTrajectoryOnline,
    to_local_frame,
)
from gpudrive.datatypes.roadgraph import (
    LocalRoadGraphPoints,
    GlobalRoadGraphPoints,
)
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
    LidarObs,
    BevObs,
)
from gpudrive.datatypes.metadata import Metadata
from gpudrive.datatypes.info import Info
from gpudrive.visualize.core import MatplotlibVisualizer
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

from gpudrive.integrations.vbd.data.utils import process_scenario_data

from gpudrive.utils.preprocess import smooth_scenario


class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        data_loader,
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
        backend="torch",
    ):
        # Initialization of environment configurations
        self.config = config
        self.data_loader = data_loader
        self.num_worlds = data_loader.batch_size
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config
        self.backend = backend
        self.max_num_agents_in_scene = self.config.max_num_agents_in_scene

        # Initialize reward weights tensor to None initially
        self.reward_weights_tensor = None

        # Environment parameter setup
        params = self._setup_environment_parameters()

        # Initialize the iterator once
        self.data_iterator = iter(self.data_loader)

        # Get the initial data batch (set of traffic scenarios)
        self.data_batch = next(self.data_iterator)

        assert self.num_worlds == len(
            self.data_batch
        ), f"Number of scenarios in data_batch ({len(self.data_batch)}) \
        should equal number of worlds ({self.num_worlds}). \
        \n Please check your data loader configuration."

        assert self.num_worlds == len(
            self.data_batch
        ), f"Number of scenarios in data_batch ({len(self.data_batch)}) \
        should equal number of worlds ({self.num_worlds}). \
        \n Please check your data loader configuration."

        # Initialize simulator
        self.sim = self._initialize_simulator(params, self.data_batch)

        self.init_steps = self.config.init_steps

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        self.episode_len = self.config.episode_len
        self.step_in_world = (
            self.episode_len - self.sim.steps_remaining_tensor().to_torch()
        )

        self.setup_guidance()

        if self.config.reward_type == "reward_conditioned":
            # Use default condition_mode from config or fall back to "random"
            condition_mode = getattr(self.config, "condition_mode", "random")
            self.agent_type = getattr(self.config, "agent_type", None)
            self._set_reward_weights(
                condition_mode=condition_mode, agent_type=self.agent_type
            )

        self.previous_action_value_tensor = torch.zeros(
            (self.num_worlds, self.max_cont_agents, 3), device=self.device
        )

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.get_obs(self.cont_agent_mask).shape[-1],),
        )

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_space.shape[-1],),
            dtype=np.float32,
        )

        # Action space setup
        # Action space setup
        self._setup_action_space(action_type)
        self.single_action_space = self.action_space
        self.num_agents = self.cont_agent_mask.sum().item()

        # Rendering setup
        self.vis = MatplotlibVisualizer(
            sim_object=self.sim,
            controlled_agent_mask=self.cont_agent_mask,
            reference_trajectory=self.reference_trajectory,
            goal_radius=self.config.dist_to_goal_threshold,
            num_worlds=self.num_worlds,
            render_config=self.render_config,
            env_config=self.config,
        )

    def setup_guidance(self):
        """Configure the reference trajectory based on the guidance mode.

        Returns:
            reference_trajectory: The reference trajectory to guide the agent's
            behavior. Shape: [num_worlds, max_agent_count, traj_len, feature_dim], where
            features are assumed to be in the following order: [x, y, yaw, vx, vy]
        """
        self.guidance_mode = self.config.guidance_mode

        if self.guidance_mode == "vbd_amortized":
            # Use pre-generated VBD model predictions as guidance
            trajectory_tensor = self.sim.vbd_trajectory_tensor()
            self.reference_trajectory = VBDTrajectory.from_tensor(
                trajectory_tensor, self.backend, self.device
            )
        elif self.guidance_mode == "vbd_online":

            self.init_steps = max(self.init_steps, 10)
            print(
                f"\n[Note] Guidance mode '{self.guidance_mode}' requires at least {self.init_steps} initialization steps to provide sufficient scene context for the diffusion model. Automatically setting simulator time to t = {self.init_steps}. \n"
            )

            # Load pre-trained Versatile Behavior Diffusion (VBD) model
            self.vbd_model = self._load_vbd_model(
                model_path=self.config.vbd_model_path
            )
            # Construct scene context dict for the VBD model
            scene_context = self.construct_context(init_steps=self.init_steps)

            print("Generating VBD predictions...\n")

            # Query the model online for the reference trajectory
            predicted = self.vbd_model.sample_denoiser(scene_context)

            # Wrap predictions into a VBDTrajectoryOnline object
            self.reference_trajectory = VBDTrajectoryOnline.from_tensor(
                vbd_predictions=predicted["denoised_trajs"],
                mean_pos_xy=self.sim.world_means_tensor().to_torch()[:, :2],
                backend=self.backend,
                device=self.device,
            )

        else:  # Default option is "log_replay"
            trajectory_tensor = self.sim.expert_trajectory_tensor()
            self.reference_trajectory = LogTrajectory.from_tensor(
                trajectory_tensor,
                self.num_worlds,
                self.max_agent_count,
                self.backend,
                self.device,
            )

        # Length of the guidance trajectory
        self.reference_traj_len = self.reference_trajectory.length

        # Smooth the trajectory if specified
        if self.config.smoothen_trajectory:
            self.reference_trajectory = smooth_scenario(
                self.reference_trajectory
            )

        # Initialize the reference trajectory positions that are already in reach
        self.guidance_points_hit, _ = self.guidance_points_within_reach()

    def _load_vbd_model(self, model_path):
        """
        Load the Versatile Behavior Diffusion (VBD) weights from checkpoint.
        """
        from gpudrive.integrations.vbd.sim_agent.sim_actor import VBDTest

        model = VBDTest.load_from_checkpoint(
            model_path, torch.device(self.device)
        )
        model.reset_agent_length(self.max_cont_agents)
        model.guidance_iter = 5
        _ = model.eval()
        return model

    def construct_context(self, init_steps=10):
        """
        Construct a dictionary containing information from the first 10 steps (1s) of the scene.

        This context data is used for the pre-trained VBD model, which infers
        the most likely trajectory for the next 80 steps based on this context.

        Args:
            init_steps (int): Number of steps to use for context. Default is 10.

        Returns:
            dict: Dictionary containing context information with the following keys:
                - 'agents_history': Past agent positions and states
                - 'agents_interested': Agents marked for trajectory prediction
                - 'agents_type': Type classification of each agent
                - 'agents_future': Ground truth future states (if available)
                - 'traffic_light_points': Location and state of traffic signals
                - 'polylines': Road network geometry representation
                - 'polylines_valid': Validity flags for polyline segments
                - 'relations': Interaction relationships between agents
                - 'agents_id': Unique identifiers for each agent
                - 'anchors': Reference points
        """
        means_xy = (
            self.sim.world_means_tensor().to_torch()[:, :2].to(self.device)
        )

        # Get the logged trajectory and restore the mean
        log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
            device=self.device,
        )
        log_trajectory.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )

        # Get global road graph and restore the mean
        global_road_graph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        global_road_graph.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )
        global_road_graph.restore_xy()

        # Get global agent observations and restore the mean
        global_agent_obs = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=self.sim.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        global_agent_obs.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )
        metadata = Metadata.from_tensor(
            metadata_tensor=self.sim.metadata_tensor(),
            backend=self.backend,
            device=self.device,
        )
        context_dict = process_scenario_data(
            max_controlled_agents=self.max_cont_agents,
            controlled_agent_mask=self.cont_agent_mask.cpu(),
            global_agent_obs=global_agent_obs,
            global_road_graph=global_road_graph,
            log_trajectory=log_trajectory,
            episode_len=self.episode_len,
            init_steps=init_steps,
            raw_agent_types=self.sim.info_tensor().to_torch()[:, :, 4],
            metadata=metadata,
        )

        return context_dict

    def _set_reward_weights(self, condition_mode="random", agent_type=None):
        """Set agent reward weights for all or specific environments.

        Args:
            condition_mode: Determines how reward weights are sampled:
                        - "random": Random sampling within bounds (default for training)
                        - "fixed": Use predefined agent_type weights (for testing)
                        - "preset": Use a specific preset from agent_type parameter
            agent_type: Specifies which preset weights to use if condition_mode is "preset" or "fixed"
                    If condition_mode is "preset", can be one of: "cautious", "aggressive", "balanced"
                    If condition_mode is "fixed", should be a tensor of shape [3] with weight values
        """
        # Use weight sharing across environments for memory efficiency
        if self.reward_weights_tensor is None:
            self.reward_weights_tensor = torch.zeros(
                self.cont_agent_mask.shape[1],  # max_agent_count from mask
                3,  # collision, goal_achieved, off_road
                device="cpu",
                dtype=torch.float16,
            )

        # Read bounds for the three reward components
        lower_bounds = torch.tensor(
            [
                self.config.collision_weight_lb,
                self.config.goal_achieved_weight_lb,
                self.config.off_road_weight_lb,
            ],
            device="cpu",
            dtype=torch.float16,
        )

        upper_bounds = torch.tensor(
            [
                self.config.collision_weight_ub,
                self.config.goal_achieved_weight_ub,
                self.config.off_road_weight_ub,
            ],
            device="cpu",
            dtype=torch.float16,
        )
        bounds_range = upper_bounds - lower_bounds

        # Preset agent personality types
        agent_presets = {
            "cautious": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.9,  # Strong collision penalty
                    self.config.goal_achieved_weight_ub
                    * 0.7,  # Moderate goal reward
                    self.config.off_road_weight_lb
                    * 0.9,  # Strong off-road penalty
                ],
                device=self.device,
                dtype=torch.float16,
            ),
            "aggressive": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.5,  # Lower collision penalty
                    self.config.goal_achieved_weight_ub
                    * 0.9,  # Higher goal reward
                    self.config.off_road_weight_lb
                    * 0.6,  # Moderate off-road penalty
                ],
                device=self.device,
                dtype=torch.float16,
            ),
            "balanced": torch.tensor(
                [
                    (
                        self.config.collision_weight_lb
                        + self.config.collision_weight_ub
                    )
                    / 2,
                    (
                        self.config.goal_achieved_weight_lb
                        + self.config.goal_achieved_weight_ub
                    )
                    / 2,
                    (
                        self.config.off_road_weight_lb
                        + self.config.off_road_weight_ub
                    )
                    / 2,
                ],
                device=self.device,
                dtype=torch.float16,
            ),
            "risk_taker": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.3,  # Minimal collision penalty
                    self.config.goal_achieved_weight_ub,  # Maximum goal reward
                    self.config.off_road_weight_lb
                    * 0.4,  # Low off-road penalty
                ],
                device=self.device,
                dtype=torch.float16,
            ),
        }
        # Just get the max agents dimension from the controlled agent mask
        max_agents = self.cont_agent_mask.shape[1]

        if condition_mode == "random":
            # Traditional random sampling within bounds
            random_values = torch.rand(
                max_agents,
                3,
                device="cpu",
                dtype=torch.float16,
            )
            scaled_values = lower_bounds + random_values * bounds_range

        elif condition_mode == "preset":
            # Use a predefined agent type
            if agent_type not in agent_presets:
                raise ValueError(
                    f"Unknown agent_type: {agent_type}. Available types: {list(agent_presets.keys())}"
                )

            # CHANGED: Create a tensor with the preset weights for all agents, but no environment dimension
            preset_weights = agent_presets[agent_type]
            scaled_values = preset_weights.unsqueeze(0).expand(max_agents, 3)

        elif condition_mode == "fixed":
            # Use custom provided weights
            if agent_type is None or not isinstance(agent_type, torch.Tensor):
                raise ValueError(
                    "For condition_mode='fixed', agent_type must be a tensor of shape [3]"
                )

            custom_weights = agent_type.to(device="cpu", dtype=torch.float16)
            if custom_weights.shape != (3,):
                raise ValueError(
                    f"agent_type tensor must have shape [3], got {custom_weights.shape}"
                )

            scaled_values = custom_weights.unsqueeze(0).expand(max_agents, 3)

        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        self.reward_weights_tensor = scaled_values

        return self.reward_weights_tensor

    def reset(
        self,
        mask=None,
        env_idx_list=None,
        condition_mode=None,
        agent_type=None,
    ):
        """Reset the worlds and return the initial observations.

        Args:
            mask: Optional mask indicating which agents to return observations for
            env_idx_list: Optional list of environment indices to reset
            condition_mode: Determines how reward weights are sampled:
                - "random": Random sampling within bounds (default for training)
                - "fixed": Use predefined agent_type weights (for testing)
                - "preset": Use a specific preset from agent_type parameter
            agent_type: Specifies which preset weights to use or custom weights

        Returns:
            obs: The initial observations.
        """
        # Reset the simulator state
        if env_idx_list is not None:
            self.sim.reset(env_idx_list)
        else:
            env_idx_list = list(range(self.num_worlds))
            self.sim.reset(env_idx_list)

        # Re-initialize reward weights if using reward_conditioned
        if (
            hasattr(self.config, "reward_type")
            and self.config.reward_type == "reward_conditioned"
        ):
            # Use the specified condition_mode or default to the config setting
            mode = (
                condition_mode
                if condition_mode is not None
                else getattr(self.config, "condition_mode", "random")
            )
            use_agent_type = (
                agent_type if agent_type is not None else self.agent_type
            )
            self._set_reward_weights(
                condition_mode=mode, agent_type=use_agent_type
            )

        # Reset smoothness tracking for reset environments
        if env_idx_list is not None:
            reset_mask = torch.zeros(
                self.num_worlds, dtype=torch.bool, device=self.device
            )
            reset_mask[torch.tensor(env_idx_list, device=self.device)] = True

            # Zero out only the reset environments
            self.previous_action_value_tensor[reset_mask] = 0.0
            # Reset the guidance points hit tensor for the reset environments
            # TODO(dc): Fix for asynchronous resets
            self.guidance_points_hit, _ = self.guidance_points_within_reach()

        else:
            self.previous_action_value_tensor.zero_()
            self.guidance_points_hit, _ = self.guidance_points_within_reach()

        # Dropout mask for guidance points
        # Assumption: all worlds are reset at the same time
        if self.config.guidance_dropout_prob > 0:
            self.guidance_dropout_mask = self.create_guidance_dropout_mask()

        return self.get_obs(mask)

    def get_dones(self):
        """
        Returns tensor indicating which agents have terminated.
        """
        terminal = (
            self.sim.done_tensor()
            .to_torch()
            .clone()
            .squeeze(dim=2)
            .to(torch.float)
        )
        return terminal.bool()

    def get_infos(self):
        """
        Returns the info tensor for the current step.
        """
        return Info.from_tensor(
            self.sim.info_tensor(),
            backend=self.backend,
            device=self.device,
        )

    def get_rewards(
        self,
        collision_weight=-0.01,
        goal_achieved_weight=1.0,
        off_road_weight=-0.01,
    ):
        """Obtain the rewards for the current step."""

        # Return the weighted combination of the reward components
        info_tensor = self.sim.info_tensor().to_torch().clone()
        off_road = info_tensor[:, :, 0].to(torch.float)

        # True if the agent is in collision with another road object
        # (i.e. a cyclist, pedestrian or vehicle)
        collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
        goal_achieved = info_tensor[:, :, 3].to(torch.float)

        if self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

        elif self.config.reward_type == "weighted_combination":
            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )
            return weighted_rewards

        elif self.config.reward_type == "reward_conditioned":
            if self.reward_weights_tensor is None:
                self._set_reward_weights()

            # Compute the weighted rewards
            collision_weights = (
                self.reward_weights_tensor[:, 0]
                .expand(self.num_worlds, -1)
                .to(self.device)
            )
            goal_weights = (
                self.reward_weights_tensor[:, 1]
                .expand(self.num_worlds, -1)
                .to(self.device)
            )
            off_road_weights = (
                self.reward_weights_tensor[:, 2]
                .expand(self.num_worlds, -1)
                .to(self.device)
            )

            weighted_rewards = (
                collision_weights * collided
                + goal_weights * goal_achieved
                + off_road_weights * off_road
            )

            return weighted_rewards

        elif self.config.reward_type == "guided_autonomy":

            step_in_world = self.step_in_world[:, 0, :].squeeze(-1)

            # 1. Get base rewards defined by user
            self.base_rewards = (
                collision_weight * collided + off_road_weight * off_road
            )

            is_valid = self.reference_trajectory.valids.squeeze(-1)

            # 2. Route guidance (where to go)
            # a) Get the reference trajectory positions
            (
                points_within_reach,
                distance_to_points,
            ) = self.guidance_points_within_reach()

            if step_in_world[0].item() == 1:
                # This is the first step, and we reward the agent for
                # reaching the initial guidance points
                new_hits = self.guidance_points_hit & is_valid
            else:
                # Find waypoints within reach that haven't been hit yet and are valid
                potential_new_hits = (
                    points_within_reach
                    & (~self.guidance_points_hit)
                    & is_valid
                ).to(self.device)

                # Create a mask to handle cases where there are no potential hits
                has_potential_hits = torch.any(
                    potential_new_hits, dim=2, keepdim=True
                )

                # Replace distances for non-potential hits with a large value
                masked_distances = torch.where(
                    potential_new_hits.bool(),
                    distance_to_points,
                    torch.ones_like(distance_to_points) * 1e10,
                ).to(self.device)

                # Find indices of closest waypoints for each agent
                closest_indices = torch.argmin(masked_distances, dim=2)

                # Create a one-hot tensor marking only the closest waypoint for each agent
                (
                    batch_size,
                    num_agents,
                    num_waypoints,
                ) = potential_new_hits.shape
                batch_indices = (
                    torch.arange(batch_size)
                    .view(-1, 1)
                    .repeat(1, num_agents)
                    .flatten()
                    .to(self.device)
                )
                agent_indices = (
                    torch.arange(num_agents).repeat(batch_size).to(self.device)
                )

                # Initialize new_hits tensor with zeros
                new_hits = torch.zeros_like(potential_new_hits)

                # Only set closest hits where there are potential hits
                valid_agents = has_potential_hits.view(-1)

                # Efficiently set the closest point for each valid agent
                if torch.any(valid_agents):
                    valid_batch_indices = batch_indices[valid_agents]
                    valid_agent_indices = agent_indices[valid_agents]
                    valid_closest_indices = closest_indices.view(-1)[
                        valid_agents
                    ]

                    new_hits[
                        valid_batch_indices,
                        valid_agent_indices,
                        valid_closest_indices,
                    ] = True

            # Update the guidance points hit tensor
            self.guidance_points_hit = self.guidance_points_hit | new_hits

            # Count the number of new waypoints hit in this step
            guidance_points_hit_count = new_hits.sum(dim=-1).float()

            # Reward agent for successfully reaching guidance points along the route
            # Scale reward by 1/num_valid_points to ensure maximum attainable reward is 1.0
            # We do this for a consistent reward magnitude regardless of route length or density
            self.route_reward = torch.clamp(
                guidance_points_hit_count
                / (self.reference_trajectory.valids.sum(axis=[2, 3]) + 1e-5),
                min=0,
                max=0.1,
            )

            # We want agents that are parked to stay parked, and generally
            # we want agents to don't go beyond their reference trajectory.
            # To do this, add a penalty proportional to the agent position to
            # the end position of the trajectory at the end of the episode
            total_valid = self.reference_trajectory.valids.clone().sum(
                axis=[2, 3]
            )
            valid_hits_so_far = self.guidance_points_hit.clone().sum(dim=[-1])

            # Handle the case where there are zero valid points
            # If total_valid is 0, we consider the route 100% complete (progress = 1.0)
            route_progress = torch.where(
                total_valid > 0,
                valid_hits_so_far / (total_valid + 1e-5),  # Normal case
                torch.ones_like(
                    valid_hits_so_far
                ),  # When agent has no valid points
            )

            completed_route_mask = (route_progress >= 0.99).float()

            # b). Add a bonus for being close to the reference end position
            if torch.any(completed_route_mask > 0):
                # Extract the last valid reference position
                agent_states = GlobalEgoState.from_tensor(
                    self.sim.absolute_self_observation_tensor(),
                    self.backend,
                    self.device,
                )

                # Get indices of last valid positions for each reference trajectory
                last_valid_indices = torch.argmax(
                    is_valid
                    * torch.arange(is_valid.shape[2], device=is_valid.device),
                    dim=2,
                )

                # Create indices for gathering
                batch_indices = torch.zeros_like(last_valid_indices)
                agent_indices = torch.arange(
                    is_valid.shape[1], device=is_valid.device
                ).expand_as(last_valid_indices)

                # Gather the last valid reference positions
                last_valid_positions = self.reference_trajectory.pos_xy[
                    batch_indices, agent_indices, last_valid_indices
                ]

                distance = torch.norm(
                    agent_states.pos_xy - last_valid_positions, dim=2
                )

                # Add bonus for being close to the reference end position
                # Using exponential decay: bonus = scale * exp(-distance)
                max_position_bonus = 0.02

                position_bonus = max_position_bonus * torch.exp(-distance)
                
                # # Apply a penalty for jerk because we don't want agents to turn around 
                # # when they have already passed the end of the route
                # if hasattr(self, "action_diff"):
                #     acceleration_jerk = (
                #         self.action_diff[:, :, 0] ** 2
                #     )  
                #     steering_jerk = (
                #         self.action_diff[:, :, 1] ** 2
                #     ) 

                #     acceleration_penalty = 1.0 - torch.exp(-acceleration_jerk)
                #     steering_penalty = 1.0 - torch.exp(-steering_jerk)
                    
                #     jerk_penalty = -0.001 * (
                #         acceleration_penalty + steering_penalty
                #     )

                position_bonus_completed = position_bonus * completed_route_mask
                
                #position_bonus_completed = (position_bonus + jerk_penalty) * completed_route_mask

                self.route_reward += position_bonus_completed 

            self.guidance_reward = self.route_reward.clone()

            # 3. Speed and heading targets
            if step_in_world[0] < self.reference_traj_len:
                batch_idx = torch.arange(step_in_world.shape[0])

                suggested_speed = self.reference_trajectory.ref_speed[
                    batch_idx, :, step_in_world
                ].squeeze(-1)

                suggested_heading = self.reference_trajectory.yaw[
                    batch_idx, :, step_in_world
                ].squeeze(-1)

                valid_points = (
                    self.reference_trajectory.valids[
                        batch_idx, :, step_in_world
                    ]
                    .squeeze(-1)
                    .bool()
                )

                actual_agent_speed = (
                    self.sim.self_observation_tensor().to_torch()[:, :, 0]
                )

                actual_agent_heading = self.agent_states.rotation_angle

                # Compute distances
                guidance_speed_error = (
                    suggested_speed - actual_agent_speed
                ) ** 2
                guidance_heading_error = (
                    suggested_heading - actual_agent_heading
                ) ** 2

                guidance_speed_penalty = 1.0 - torch.exp(
                    -guidance_speed_error + 1e-8
                )
                guidance_heading_penalty = 1.0 - torch.exp(
                    -guidance_heading_error + 1e-8
                )

                speed_heading_reward = (
                    -self.config.guidance_speed_weight * guidance_speed_penalty
                    - self.config.guidance_heading_weight
                    * guidance_heading_penalty
                )

                speed_heading_reward[~valid_points] = 0.0

                self.speed_heading_reward = speed_heading_reward.clone()

                self.guidance_reward += self.speed_heading_reward

            # 4. Penalty for action jerk
            if hasattr(self, "action_diff"):
                acceleration_jerk = (
                    self.action_diff[:, :, 0] ** 2
                )  # First action component is acceleration
                steering_jerk = (
                    self.action_diff[:, :, 1] ** 2
                )  # Second action component is steering angle

                # Small jerks: penalty is close to x (approximately linear)
                # Large jerks: penalty approaches 1.0 (saturates)
                acceleration_penalty = 1.0 - torch.exp(-acceleration_jerk)
                steering_penalty = 1.0 - torch.exp(-steering_jerk)

                self.smoothness_penalty = -self.config.smoothness_weight * (
                    acceleration_penalty + steering_penalty
                )
            else:
                self.smoothness_penalty = torch.zeros_like(self.base_rewards)

            self.guidance_reward += self.smoothness_penalty

            # Combine
            rewards = self.base_rewards + self.guidance_reward

            return rewards

    def create_guidance_dropout_mask(self):
        """
        Create guidance dropout mask where dropout_prob represents 
        the maximum dropout probability, with varying rates per trajectory.
        """
        max_dropout_prob = self.config.guidance_dropout_prob
        num_controlled = self.cont_agent_mask.sum().item()
        
        # 1 if we want to keep the point, 0 if we want to drop it
        guidance_dropout_mask = torch.ones(
            (num_controlled, self.reference_traj_len),
            device=self.device,
            dtype=torch.bool,
        )

        is_valid = (
            self.reference_trajectory.valids[self.cont_agent_mask]
            .squeeze(-1)
            .bool()
        )
        
        # Generate random dropout rates for each agent between 0 and max_dropout_prob
        agent_dropout_probs = torch.rand(num_controlled, device=self.device) * max_dropout_prob
        
        for agent_idx in range(num_controlled):
            agent_valid_mask = is_valid[agent_idx]
            agent_valid_indices = torch.where(agent_valid_mask)[0]
            
            # Get agent-specific dropout probability
            agent_dropout_prob = agent_dropout_probs[agent_idx]

            if len(agent_valid_indices) > 2:  # Only apply dropout if we have more than 2 points
                # Keep first and last points, apply dropout to middle points
                middle_indices = agent_valid_indices[1:-1]

                # Generate random dropout mask for middle points
                dropout = (
                    torch.rand(len(middle_indices), device=self.device)
                    < agent_dropout_prob
                )

                # Apply dropout to middle points (set to False for points to drop)
                guidance_dropout_mask[agent_idx, middle_indices] = ~dropout
        
        return guidance_dropout_mask

    def guidance_points_within_reach(self):
        # Get actual agent positions
        self.agent_states = GlobalEgoState.from_tensor(
            self.sim.absolute_self_observation_tensor(),
            self.backend,
            self.device,
        )

        # Calculate Euclidean distance to all reference positions
        # Output shape: [worlds, agents, reference_traj_len]
        distances = torch.norm(
            self.agent_states.pos_xy.unsqueeze(2)
            - self.reference_trajectory.pos_xy,
            dim=-1,
        )

        points_within_reach = (
            distances < self.config.dist_to_goal_threshold
        ) & self.reference_trajectory.valids.bool().squeeze(-1)

        return points_within_reach, distances

    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

        # Update time in worlds
        self.step_in_world = (
            self.episode_len - self.sim.steps_remaining_tensor().to_torch()
        )

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
            or self.config.dynamics_model == "delta_local"
        ):
            if actions.dim() == 2:  # (num_worlds, max_agent_count)
                # Map action indices to action values if indices are provided
                actions = (
                    torch.nan_to_num(actions, nan=0).long().to(self.device)
                )
                self.action_value_tensor = self.action_keys_tensor[actions]

            elif actions.dim() == 3:
                if actions.shape[2] == 1:
                    actions = actions.squeeze(dim=2).to(self.device)
                    self.action_value_tensor = self.action_keys_tensor[actions]
                else:  # Assuming we are given the actual action values
                    self.action_value_tensor = actions.to(self.device)
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")

        else:
            self.action_value_tensor = actions.to(self.device)

        if not hasattr(self, "previous_action_value_tensor"):
            # Initialize with current actions on first call
            self.previous_action_value_tensor = (
                self.action_value_tensor.clone()
            )

        if (
            self.config.dynamics_model == "state"
            and self.previous_action_value_tensor.shape
            != self.action_value_tensor.shape
        ):
            self.previous_action_value_tensor = (
                self.action_value_tensor.clone()
            )

        # Calculate action differences (jerk)
        self.action_diff = (
            self.action_value_tensor - self.previous_action_value_tensor
        )
        self.previous_action_value_tensor = self.action_value_tensor.clone()

        self._copy_actions_to_simulator(self.action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provided actions to the simulator."""
        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            # Action space: (acceleration, steering, heading)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "delta_local":
            # Action space: (dx, dy, dyaw)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "state":
            # Following the StateAction struct in types.hpp
            # Need to provide:
            # (x, y, z, yaw, vel x, vel y, vel z, ang_vel_x, ang_vel_y, ang_vel_z)
            self.sim.action_tensor().to_torch()[:, :, :10].copy_(actions)
        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space based on dynamics model."""
        products = None

        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            products = product(self.dx, self.dy, self.dyaw)
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = self.config.head_tilt_actions.to(self.device)
            products = product(
                self.accel_actions, self.steer_actions, self.head_actions
            )
        elif self.config.dynamics_model == "state":
            self.x = self.config.x.to(self.device)
            self.y = self.config.y.to(self.device)
            self.yaw = self.config.yaw.to(self.device)
            self.vx = self.config.vx.to(self.device)
            self.vy = self.config.vy.to(self.device)

        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
        self.values_to_action_key = {}
        if products is not None:
            for action_idx, (action_1, action_2, action_3) in enumerate(
                products
            ):
                self.action_key_to_values[action_idx] = [
                    action_1.item(),
                    action_2.item(),
                    action_3.item(),
                ]
                self.values_to_action_key[
                    round(action_1.item(), 5),
                    round(action_2.item(), 5),
                    round(action_3.item(), 5),
                ] = action_idx

            self.action_keys_tensor = torch.tensor(
                [
                    self.action_key_to_values[key]
                    for key in sorted(self.action_key_to_values.keys())
                ]
            ).to(self.device)

            return Discrete(n=int(len(self.action_key_to_values)))
        else:
            return Discrete(n=1)

    def _set_continuous_action_space(self) -> None:
        """Configure the continuous action space."""
        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            action_1 = self.dx.clone().cpu().numpy()
            action_2 = self.dy.clone().cpu().numpy()
            action_3 = self.dyaw.clone().cpu().numpy()
        elif self.config.dynamics_model == "classic":
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_1 = self.steer_actions.clone().cpu().numpy()
            action_2 = self.accel_actions.clone().cpu().numpy()
            action_3 = self.head_actions.clone().cpu().numpy()
        else:
            raise ValueError(
                f"Continuous action space is currently not supported for dynamics_model: {self.config.dynamics_model}."
            )

        action_space = Tuple(
            (
                Box(action_1.min(), action_1.max(), shape=(1,)),
                Box(action_2.min(), action_2.max(), shape=(1,)),
                Box(action_3.min(), action_3.max(), shape=(1,)),
            )
        )
        return action_space

    def _get_guidance(self, mask=None) -> torch.Tensor:
        """Receive (expert) suggestions from pre-trained model or logs."""

        if not self.config.guidance:
            return torch.zeros(0, device=self.device)

        guidance = []
        guidance_orig = []

        if mask is None:

            valid_timesteps_mask = self.reference_trajectory.valids.bool()

            if self.config.add_reference_speed:
                reference_speed = (
                    self.reference_trajectory.ref_speed.clone()
                    / constants.MAX_SPEED
                )
                reference_speed[~valid_timesteps_mask] = constants.INVALID_ID
                guidance.append(reference_speed)

            states = None
            if (
                self.config.add_reference_pos_xy
                or self.config.add_reference_heading
            ):
                states = GlobalEgoState.from_tensor(
                    self.sim.absolute_self_observation_tensor(),
                    self.backend,
                    self.device,
                )

            if self.config.add_reference_pos_xy:
                glob_reference_xy = self.reference_trajectory.pos_xy
                local_reference_xy = torch.empty_like(glob_reference_xy)

                # Transform reference path to be relative to current
                # agent positions and heading
                for world_idx in range(self.num_worlds):
                    for agent_idx in range(self.max_cont_agents):
                        local_reference_xy[
                            world_idx, agent_idx, :, :
                        ] = to_local_frame(
                            global_pos_xy=glob_reference_xy[
                                world_idx, agent_idx, :, :
                            ],
                            ego_pos=states.pos_xy[world_idx, agent_idx],
                            ego_yaw=states.rotation_angle[
                                world_idx, agent_idx
                            ],
                            device=self.device,
                        )

                local_ref_xy_orig = local_reference_xy.clone()

                # Normalize
                local_reference_xy /= constants.MAX_REF_POINT

                # Set invalid steps to -1.0
                local_reference_xy[
                    ~valid_timesteps_mask.expand_as(local_reference_xy)
                ] = constants.INVALID_ID

                # Make unnormalized reference path available for plotting
                self.reference_path = local_ref_xy_orig

                guidance.append(local_reference_xy)

            if self.config.add_reference_heading:
                reference_headings = self.reference_trajectory.yaw.clone()

                # Transform headings to local coordinate frame
                for world_idx in range(self.num_worlds):
                    for agent_idx in range(self.max_cont_agents):
                        # Subtract current agent heading to get relative heading
                        reference_headings[
                            world_idx, agent_idx
                        ] -= states.rotation_angle[world_idx, agent_idx]

                # Normalize
                reference_headings = (
                    reference_headings / constants.MAX_ORIENTATION_RAD
                )

                # Set invalid timesteps to -1
                reference_headings[
                    ~valid_timesteps_mask
                ] = constants.INVALID_ID
                guidance.append(reference_headings)

            return torch.cat(guidance, dim=-1).flatten(start_dim=2)

        else:
            valid_timesteps_mask = self.reference_trajectory.valids.bool()[
                mask
            ]

            states = None
            if (
                self.config.add_reference_pos_xy
                or self.config.add_reference_heading
            ):
                states = GlobalEgoState.from_tensor(
                    self.sim.absolute_self_observation_tensor(),
                    self.backend,
                    self.device,
                )

            if self.config.add_reference_speed:
                reference_speed = self.reference_trajectory.ref_speed[mask].clone()
                reference_speed[~valid_timesteps_mask] = constants.INVALID_ID
                
                reference_speed_normalized = reference_speed / constants.MAX_SPEED
                guidance_orig.append(reference_speed)
                guidance.append(reference_speed_normalized)

            if self.config.add_reference_pos_xy:
                global_reference_xy = self.reference_trajectory.pos_xy.clone()[
                    mask
                ]

                # Translate all points to a local coordinate frame
                translated = global_reference_xy - states.pos_xy[
                    mask
                ].unsqueeze(1)

                # Create batch of rotation matrices: [batch, 2, 2]
                cos_yaw = torch.cos(states.rotation_angle[mask])
                sin_yaw = torch.sin(states.rotation_angle[mask])
                rotation_matrices = torch.stack(
                    [
                        torch.stack([cos_yaw, sin_yaw], dim=1),
                        torch.stack([-sin_yaw, cos_yaw], dim=1),
                    ],
                    dim=1,
                )

                # Apply rotation to all points
                local_reference_xy = torch.bmm(
                    rotation_matrices, translated.transpose(1, 2)
                ).transpose(1, 2)

                local_reference_xy_orig = local_reference_xy.clone()

                # Normalize to [-1, 1]
                local_reference_xy /= constants.MAX_REF_POINT

                # Set invalid timesteps to -1
                local_reference_xy[
                    ~valid_timesteps_mask.expand_as(local_reference_xy)
                ] = constants.INVALID_ID

                reference_path = local_reference_xy

                self.reference_path = local_reference_xy_orig

                guidance.append(reference_path)
                guidance_orig.append(local_reference_xy_orig)

            if self.config.add_reference_heading:
                reference_headings = self.reference_trajectory.yaw[
                    mask
                ].clone()

                # Translate headings to local coordinate frame by subtracting current global agent headings
                reference_headings = (
                    reference_headings
                    - states.rotation_angle[mask].view(-1, 1, 1)
                )

                # Normalize by 2pi to ensure values are in [-1, 1]
                reference_headings_normalized = (
                    reference_headings / constants.MAX_ORIENTATION_RAD
                )

                # Set invalid timesteps to -1
                reference_headings[
                    ~valid_timesteps_mask
                ] = constants.INVALID_ID
                guidance.append(reference_headings_normalized)
                guidance_orig.append(reference_headings)

        self.guidance_obs = torch.cat(guidance_orig, dim=-1)

        # Apply dropout mask if specified
        # Note: currently only supported for masked observations
        if self.config.guidance_dropout_prob > 0 and hasattr(
            self, "guidance_dropout_mask"
        ):
            self.guidance_obs[
                ~self.guidance_dropout_mask
            ] = constants.INVALID_ID
            self.reference_path[
                ~self.guidance_dropout_mask
            ] = constants.INVALID_ID

        self.valid_guidance_points = torch.sum(
            self.guidance_obs[:, :, 0] != constants.INVALID_ID, axis=1
        )
        
        guidance = torch.cat(guidance, dim=-1)

        return guidance.flatten(start_dim=1)

    def _get_ego_state(self, mask=None) -> torch.Tensor:
        """Get the ego state."""

        if not self.config.ego_state:
            return torch.Tensor().to(self.device)

        ego_state = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim.self_observation_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )

        if self.config.norm_obs:
            ego_state.normalize()

        base_fields = [
            ego_state.speed.unsqueeze(-1),
            ego_state.vehicle_length.unsqueeze(-1),
            ego_state.vehicle_width.unsqueeze(-1),
            ego_state.steer_angle.unsqueeze(-1),
        ]

        if mask is None:

            # New: Give agent sense of progress / history
            # 1. Overall trajectory progress (percentage of valid points hit)
            total_valid = self.reference_trajectory.valids.clone().sum(
                axis=[2, 3]
            )
            valid_hits_so_far = self.guidance_points_hit.clone().sum(dim=[-1])

            # If total_valid is 0, we consider the route 100% complete (progress = 1.0)
            self.route_progress = torch.where(
                total_valid > 0,
                valid_hits_so_far / (total_valid + 1e-5),  # Normal case
                torch.ones_like(
                    valid_hits_so_far
                ),  # When agent has no valid points
            )

            # 2. How much time do I have left to complete the route
            normalized_time = self.step_in_world.clone() / self.episode_len

            base_fields.append(self.route_progress.unsqueeze(-1))
            base_fields.append(normalized_time)

            if self.config.add_previous_action:
                normalized_prev_actions = (
                    self.previous_action_value_tensor[:, :, :2]
                    / constants.MAX_ACTION_VALUE
                )
                base_fields.append(normalized_prev_actions)

            if self.config.reward_type == "reward_conditioned":
                full_fields = base_fields + [
                    self.reward_weights_tensor.expand(self.num_worlds, -1)
                ]
                return torch.stack(full_fields).permute(1, 2, 0)
            else:
                return torch.cat(base_fields, dim=-1)
        else:

            # New: Give agent sense of progress / history
            # 1. Overall trajectory progress (percentage of valid points hit)
            total_valid = (
                self.reference_trajectory.valids[mask]
                .clone()
                .sum(dim=1)
                .squeeze(-1)
            )
            valid_hits_so_far = self.guidance_points_hit.clone()[mask].sum(
                dim=-1
            )

            # Handle the case where there are zero valid points
            # If total_valid is 0, we consider the route 100% complete (progress = 1.0)
            self.route_progress = torch.where(
                total_valid >= 1,
                valid_hits_so_far / (total_valid + 1e-5),  # Normal case
                torch.ones_like(
                    valid_hits_so_far
                ),  # When agent has no valid points
            )

            # 2. How much time do I have left
            normalized_time = self.step_in_world[mask] / self.episode_len

            base_fields.append(self.route_progress.unsqueeze(-1))
            base_fields.append(normalized_time)

            if self.config.add_previous_action:
                normalized_prev_actions = (
                    self.previous_action_value_tensor[:, :, :2][mask]
                    / constants.MAX_ACTION_VALUE
                )
                base_fields.append(normalized_prev_actions)

            if self.config.reward_type == "reward_conditioned":
                _, agent_indices = torch.where(mask)
                weights_for_masked_agents = self.reward_weights_tensor.to(
                    self.device
                )[agent_indices]

                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                        weights_for_masked_agents[:, 0],
                        weights_for_masked_agents[:, 1],
                        weights_for_masked_agents[:, 2],
                    ]
                ).permute(1, 0)
            else:                   
                return torch.cat(base_fields, dim=1)

    def _get_partner_obs(self, mask=None):
        """Get partner observations."""

        if not self.config.partner_obs:
            return torch.Tensor().to(self.device)

        partner_obs = PartnerObs.from_tensor(
            partner_obs_tensor=self.sim.partner_observations_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )

        if self.config.norm_obs:
            partner_obs.normalize()

        if mask is not None:
            return partner_obs.data.flatten(start_dim=1)
        else:
            return torch.concat(
                [
                    partner_obs.speed,
                    partner_obs.rel_pos_x,
                    partner_obs.rel_pos_y,
                    partner_obs.orientation,
                    partner_obs.vehicle_length,
                    partner_obs.vehicle_width,
                ],
                dim=-1,
            ).flatten(start_dim=2)

    def _get_road_map_obs(self, mask=None):
        """Get road map observations."""
        if not self.config.road_map_obs:
            return torch.Tensor().to(self.device)

        roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=self.sim.agent_roadmap_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )
        roadgraph.one_hot_encode_road_point_types()

        if self.config.norm_obs:
            roadgraph.normalize()

        if mask is not None:
            return torch.cat(
                [
                    roadgraph.data,
                    roadgraph.type,
                ],
                dim=-1,
            ).flatten(start_dim=1)
        else:
            return torch.cat(
                [
                    roadgraph.x.unsqueeze(-1),
                    roadgraph.y.unsqueeze(-1),
                    roadgraph.segment_length.unsqueeze(-1),
                    roadgraph.segment_width.unsqueeze(-1),
                    roadgraph.segment_height.unsqueeze(-1),
                    roadgraph.orientation.unsqueeze(-1),
                    roadgraph.type,
                ],
                dim=-1,
            ).flatten(start_dim=2)

    def _get_lidar_obs(self, mask=None):
        """Get lidar observations."""

        if not self.config.lidar_obs:
            return torch.Tensor().to(self.device)

        lidar = LidarObs.from_tensor(
            lidar_tensor=self.sim.lidar_tensor(),
            backend=self.backend,
            device=self.device,
        )

        if mask is not None:
            return [
                lidar.agent_samples[mask],
                lidar.road_edge_samples[mask],
                lidar.road_line_samples[mask],
            ]
        else:
            return torch.cat(
                [
                    lidar.agent_samples,
                    lidar.road_edge_samples,
                    lidar.road_line_samples,
                ],
                dim=-1,
            ).flatten(start_dim=2)

    def _get_bev_obs(self, mask=None):
        """Get BEV segmentation map observation.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, resolution, resolution, 1)
        """
        if not self.config.bev_obs:
            return torch.Tensor().to(self.device)

        bev = BevObs.from_tensor(
            bev_tensor=self.sim.bev_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        bev.one_hot_encode_bev_map()

        if mask is not None:
            return bev.bev_segmentation_map[mask].flatten(start_dim=1)
        else:
            return bev.bev_segmentation_map.flatten(start_dim=2)

    def get_obs(self, mask=None):
        """Get observation: Combine different types of environment information into a single tensor.
        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """
        ego_states = self._get_ego_state(mask)
        partner_observations = self._get_partner_obs(mask)
        road_map_observations = self._get_road_map_obs(mask)
        guidance_obs = self._get_guidance(mask)

        obs = torch.cat(
            (
                ego_states,
                partner_observations,
                road_map_observations,
                guidance_obs,
            ),
            dim=-1,
        )
    
        return obs

    def get_controlled_agents_mask(self):
        """Get the control mask. Shape: [num_worlds, max_agent_count]"""
        return (
            self.sim.controlled_state_tensor().to_torch().clone() == 1
        ).squeeze(axis=2)

    def remove_agents_by_id(
        self, perc_to_rmv_per_scene, remove_controlled_agents=True
    ):
        """Delete random agents in scenarios.

        Args:
            perc_to_rmv_per_scene (float): Percentage of agents to remove per scene
            remove_controlled_agents (bool): If True, removes controlled agents. If False, removes uncontrolled agents
        """
        # Obtain agent ids
        agent_ids = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim.self_observation_tensor(),
            backend="torch",
            device=self.device,
        ).id

        # Choose the appropriate mask based on whether we're removing controlled or uncontrolled agents
        if remove_controlled_agents:
            agent_mask = self.cont_agent_mask
        else:
            # Create inverse mask for uncontrolled agents
            agent_mask = ~self.cont_agent_mask

        for env_idx in range(self.num_worlds):
            # Get all relevant agent IDs (controlled or uncontrolled) for the current environment
            scene_agent_ids = agent_ids[env_idx, :][agent_mask[env_idx]].long()

            if (
                scene_agent_ids.numel() > 0
            ):  # Ensure there are agents to sample
                # Determine the number of agents to sample (X% of the total agents)
                num_to_sample = max(
                    1, int(perc_to_rmv_per_scene * scene_agent_ids.size(0))
                )

                # Randomly sample agent IDs to remove using torch
                sampled_indices = torch.randperm(scene_agent_ids.size(0))[
                    :num_to_sample
                ]
                sampled_agent_ids = scene_agent_ids[sampled_indices]

                # Delete the sampled agents from the environment
                self.sim.deleteAgents({env_idx: sampled_agent_ids.tolist()})

        # Reset controlled agent mask and visualizer
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(
            controlled_agent_mask=self.cont_agent_mask,
            reference_trajectory=self.reference_trajectory,
        )

    def swap_data_batch(self, data_batch=None):
        """
        Swap the current data batch in the simulator with a new one
        and reinitialize dependent attributes.
        """

        if data_batch is None:  # Sample new data batch from the data loader
            self.data_batch = next(self.data_iterator)
        else:
            self.data_batch = data_batch

        # Validate that the number of worlds (envs) matches the batch size
        if len(self.data_batch) != self.num_worlds:
            raise ValueError(
                f"Data batch size ({len(self.data_batch)}) does not match "
                f"the expected number of worlds ({self.num_worlds})."
            )

        # Update the simulator with the new data
        self.sim.set_maps(self.data_batch)

        # Reinitialize the mask for controlled agents
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Receive guidance trajectories from the new batch of scenarios
        self.setup_guidance()

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(
            controlled_agent_mask=self.cont_agent_mask,
            reference_trajectory=self.reference_trajectory,
        )

    def get_expert_actions(self):
        """Get expert actions for the full trajectories across worlds.

        Returns:
            expert_actions: Inferred or logged actions for the agents.
            expert_speeds: Speeds from the logged trajectories.
            expert_positions: Positions from the logged trajectories.
            expert_yaws: Heading from the logged trajectories.
        """

        log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
            device=self.device,
        )

        if self.config.dynamics_model == "delta_local":
            inferred_actions = log_trajectory.inferred_actions[..., :3]
            inferred_actions[..., 0] = torch.clamp(
                inferred_actions[..., 0], -6, 6
            )
            inferred_actions[..., 1] = torch.clamp(
                inferred_actions[..., 1], -6, 6
            )
            inferred_actions[..., 2] = torch.clamp(
                inferred_actions[..., 2], -torch.pi, torch.pi
            )
        elif self.config.dynamics_model == "state":
            # Extract (x, y, yaw, velocity x, velocity y)
            inferred_actions = torch.cat(
                (
                    log_trajectory.pos_xy,
                    torch.ones(
                        (*log_trajectory.pos_xy.shape[:-1], 1),
                        device=self.device,
                    ),
                    log_trajectory.yaw,
                    log_trajectory.vel_xy,
                    torch.zeros(
                        (*log_trajectory.pos_xy.shape[:-1], 4),
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            inferred_actions = log_trajectory.inferred_actions[..., :3]
            inferred_actions[..., 0] = torch.clamp(
                inferred_actions[..., 0], -6, 6
            )
            inferred_actions[..., 1] = torch.clamp(
                inferred_actions[..., 1], -0.3, 0.3
            )

        return (
            inferred_actions,
            log_trajectory.pos_xy,
            log_trajectory.vel_xy,
            log_trajectory.yaw,
        )

    def get_env_filenames(self):
        """Obtain the tfrecord filename for each world, mapping world indices to map names."""

        map_name_integers = self.sim.map_name_tensor().to_torch()
        filenames = {}
        # Iterate through the number of worlds
        for i in range(self.num_worlds):
            tensor = map_name_integers[i]
            # Convert ints to characters, ignoring zeros
            map_name = "".join([chr(i) for i in tensor.tolist() if i != 0])
            filenames[i] = map_name

        return filenames

    def get_scenario_ids(self):
        """Obtain the scenario ID for each world."""
        scenario_id_integers = self.sim.scenario_id_tensor().to_torch()
        scenario_ids = {}

        # Iterate through the number of worlds
        for i in range(self.num_worlds):
            tensor = scenario_id_integers[i]
            # Convert ints to characters, ignoring zeros
            scenario_id = "".join([chr(i) for i in tensor.tolist() if i != 0])
            scenario_ids[i] = scenario_id

        return scenario_ids

    def render(self, focus_env_idx=0, focus_agent_idx=[0, 1]):
        """Quick rendering function for debugging."""

        sim_states = self.vis.plot_simulator_state(
            env_indices=[focus_env_idx],
            zoom_radius=70,
            time_steps=[self.step_in_world[0, 0, 0].item()],
            plot_guidance_pos_xy=True,
        )

        agent_views = []
        for agent_idx in focus_agent_idx:
            agent_obs = self.vis.plot_agent_observation(
                env_idx=focus_env_idx,
                agent_idx=agent_idx,
                figsize=(10, 10),
                trajectory=self.reference_path[agent_idx, :, :],
                step_reward=self.guidance_reward[
                    focus_env_idx, agent_idx
                ].item(),
                route_progress=self.route_progress[agent_idx],
            )
            agent_views.append(agent_obs)

        return sim_states, agent_views

    def render(self, focus_env_idx=0, focus_agent_idx=[0, 1]):
        """Quick rendering function for debugging."""

        sim_states = self.vis.plot_simulator_state(
            env_indices=[focus_env_idx],
            zoom_radius=70,
            time_steps=[self.step_in_world[0, 0, 0].item()],
            plot_guidance_pos_xy=True,
        )

        agent_views = []
        for agent_idx in focus_agent_idx:
            agent_obs = self.vis.plot_agent_observation(
                env_idx=focus_env_idx,
                agent_idx=agent_idx,
                figsize=(10, 10),
                trajectory=self.reference_path[agent_idx, :, :],
                step_reward=self.guidance_reward[
                    focus_env_idx, agent_idx
                ].item(),
                route_progress=self.route_progress[agent_idx],
            )
            agent_views.append(agent_obs)

        return sim_states, agent_views


if __name__ == "__main__":

    FOCUS_AGENTS = [0, 1, 2, 3, 4]

    env_config = EnvConfig(
        guidance=True,
        guidance_mode="log_replay",  # Options: "log_replay", "vbd_amortized"
        add_reference_pos_xy=True,
        add_reference_speed=False,
        add_reference_heading=False,
        reward_type="guided_autonomy",
        init_mode="wosac_train",
        dynamics_model="delta_local",  # "state", #"classic",
        smoothen_trajectory=False,
        add_previous_action=True,
        guidance_dropout_prob=0.9,  # 0.95,
        
    )
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/wosac/validation_json_100",
        batch_size=1,
        dataset_size=100,
        sample_with_replacement=False,
        shuffle=False,
        file_prefix="",
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=32,  # Number of agents to control
        device="cpu",
    )

    control_mask = env.cont_agent_mask

    print(f"Number of controlled agents: {control_mask.sum()}")

    # Rollout
    obs = env.reset(mask=control_mask)

    sim_frames = []
    agent_obs_frames = {i: [] for i in FOCUS_AGENTS}

    expert_actions, _, _, _ = env.get_expert_actions()

    for time_step in range(env.init_steps, env.episode_len):
        print(f"Step: {env.step_in_world[0, 0, 0].item()}")

        # Step the environment
        expert_actions, _, _, _ = env.get_expert_actions()
        env.step_dynamics(expert_actions[:, :, time_step, :])

        obs = env.get_obs(control_mask)
        reward = env.get_rewards()
        if time_step % 10 == 0 or time_step > env.episode_len - 3:
            sim_states, agent_obs = env.render(focus_agent_idx=FOCUS_AGENTS)
            sim_frames.append(img_from_fig(sim_states[0]))
            for i in FOCUS_AGENTS:
                agent_obs_frames[i].append(img_from_fig(agent_obs[i]))

        print(f"R_t+1: {reward[0, 0]}")

        done = env.get_dones()
        info = env.get_infos()

    env.close()

    media.write_video(
        "sim_video.gif", np.array(sim_frames), fps=5, codec="gif"
    )
    for focus_agent_idx in FOCUS_AGENTS:
        agent_obs_frames[focus_agent_idx] = np.array(
            agent_obs_frames[focus_agent_idx]
        )
        media.write_video(
            f"obs_video_env_{0}_agent_{focus_agent_idx}.gif",
            np.array(agent_obs_frames[focus_agent_idx]),
            fps=5,
            codec="gif",
        )
