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
    to_local_frame,
    VBDTrajectory,
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
from gpudrive.utils.geometry import normalize_min_max

from gpudrive.integrations.vbd.data.utils import process_scenario_data


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

        self.world_time_steps = torch.zeros(
            self.num_worlds, dtype=torch.short, device=self.device
        )

        # Initialize reward weights tensor to None initially
        self.reward_weights_tensor = None

        # Environment parameter setup
        params = self._setup_environment_parameters()

        # Initialize the iterator once
        self.data_iterator = iter(self.data_loader)

        # Get the initial data batch (set of traffic scenarios)
        self.data_batch = next(self.data_iterator)

        # Initialize simulator
        self.sim = self._initialize_simulator(params, self.data_batch)

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        self.log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
            device=self.device
        )
        self.episode_len = self.config.episode_len
        self.reference_path_length = (
            self.log_trajectory.pos_xy.shape[2]
        )
        self.step_in_world = (
            self.episode_len - self.sim.steps_remaining_tensor().to_torch()
        )

        # Now initialize reward weights tensor if using reward_conditioned reward type
        if (
            hasattr(self.config, "reward_type")
            and self.config.reward_type == "reward_conditioned"
        ):
            # Use default condition_mode from config or fall back to "random"
            condition_mode = getattr(self.config, "condition_mode", "random")
            self.agent_type = getattr(self.config, "agent_type", None)
            self._set_reward_weights(
                condition_mode=condition_mode, agent_type=self.agent_type
            )

        self.previous_action_value_tensor = torch.zeros(
            (self.num_worlds, self.max_cont_agents, 3), device=self.device
        )

        # Initialize VBD model if used
        self._initialize_vbd()

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

        self._setup_action_space(action_type)
        self.single_action_space = self.action_space
        self.num_agents = self.cont_agent_mask.sum().item()

        # Rendering setup
        self.vis = MatplotlibVisualizer(
            sim_object=self.sim,
            controlled_agent_mask=self.cont_agent_mask,
            goal_radius=self.config.dist_to_goal_threshold,
            backend=self.backend,
            num_worlds=self.num_worlds,
            render_config=self.render_config,
            env_config=self.config,
        )

    def _initialize_vbd(self):
        """
        Initialize the Versatile Behavior Diffusion (VBD) model and related
        components. Link: https://arxiv.org/abs/2404.02524.
        When using amortized VBD, we don't need to run the model at runtime.

        Args:
            config: Configuration object containing VBD settings.
        """
        self.use_vbd = self.config.use_vbd
        self.vbd_trajectory_weight = self.config.vbd_trajectory_weight

        # Set initialization steps - ensure minimum steps for VBD
        if self.use_vbd:
            self.init_steps = max(
                self.config.init_steps, 11
            )  # Minimum 11 steps for VBD
        else:
            self.init_steps = self.config.init_steps

            self.vbd_trajectories = None

    def _generate_sample_batch(self, init_steps=10):
        """Generate a sample batch for the VBD model."""
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
        sample_batch = process_scenario_data(
            max_controlled_agents=self.max_cont_agents,
            controlled_agent_mask=self.cont_agent_mask,
            global_agent_obs=global_agent_obs,
            global_road_graph=global_road_graph,
            log_trajectory=log_trajectory,
            episode_len=self.episode_len,
            init_steps=init_steps,
            raw_agent_types=self.sim.info_tensor().to_torch()[:, :, 4],
            metadata=metadata,
        )
        return sample_batch

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
        """
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

        self.world_time_steps.zero_()

        # Reset smoothness tracking for reset environments
        if env_idx_list is not None:
            reset_mask = torch.zeros(
                self.num_worlds, dtype=torch.bool, device=self.device
            )
            reset_mask[torch.tensor(env_idx_list, device=self.device)] = True

            # Zero out only the reset environments
            self.previous_action_value_tensor[reset_mask] = 0.0
        else:
            self.previous_action_value_tensor.zero_()

        # Advance the simulator with log playback if warmup steps are provided
        if self.init_steps > 0:
            self.advance_sim_with_log_playback(
                init_steps=self.init_steps,
                # render_init=self.render_config.render_init,
            )

        return self.get_obs(mask)

    def get_dones(self, world_time_steps=None):
        """
        Returns tensor indicating which agents have terminated.

        Args:
            world_time_steps: Optional tensor [num_worlds] with current timestep per world.

        Returns:
            torch.Tensor: Boolean tensor [num_worlds, num_agents] where True indicates done.
        """
        terminal = (
            self.sim.done_tensor()
            .to_torch()
            .clone()
            .squeeze(dim=2)
            .to(torch.float)
        )

        if (
            world_time_steps is not None
            and self.config.reward_type == "follow_waypoints"
            and self.config.waypoint_distance_scale > 0.0
        ):
            # Find last valid timestep for each agent, this is the ground-truth episode length
            agent_episode_length = 90 - torch.argmax(
                self.log_trajectory.valids.squeeze(-1).flip(2), dim=2
            )

            expanded_time_steps = world_time_steps.unsqueeze(1).expand_as(
                agent_episode_length
            )
            return terminal.bool() & (
                expanded_time_steps >= agent_episode_length
            )

        else:
            return terminal.bool()

    def get_infos(self):
        return Info.from_tensor(
            self.sim.info_tensor(),
            backend=self.backend,
            device=self.device,
        )

    def get_rewards(
        self,
        collision_weight=-0.5,
        goal_achieved_weight=0.0,
        off_road_weight=-0.5,
    ):
        """Obtain the rewards for the current step."""

        # Return the weighted combination of the reward components
        info_tensor = self.sim.info_tensor().to_torch().clone()
        off_road = info_tensor[:, :, 0].to(torch.float)

        # True if the vehicle is in collision with another road object
        # (i.e. a cyclist or pedestrian)
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

        elif self.config.reward_type == "distance_to_vdb_trajs":
            # Reward based on distance to VBD predicted trajectories
            # (i.e. the deviation from the predicted trajectory)
            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            agent_states = GlobalEgoState.from_tensor(
                self.sim.absolute_self_observation_tensor(),
                self.backend,
                self.device,
            )

            agent_pos = torch.stack(
                [agent_states.pos_x, agent_states.pos_y], dim=-1
            )

            # Extract VBD positions at current time steps for each world
            vbd_pos = []
            for i in range(self.num_worlds):
                current_time = (
                    self.world_time_steps[i].item() - self.init_steps
                )
                # Make sure we don't exceed trajectory length
                current_time = min(
                    current_time, self.vbd_trajectories.shape[2] - 1
                )
                vbd_pos.append(self.vbd_trajectories[i, :, current_time, :2])
            vbd_pos_tensor = torch.stack(vbd_pos)

            # Compute euclidean distance between agent and logs
            dist_to_vbd = torch.norm(vbd_pos_tensor - agent_pos, dim=-1)

            # Add reward based on inverse distance to logs
            weighted_rewards += self.vbd_trajectory_weight * torch.exp(
                -dist_to_vbd
            )

            return weighted_rewards

        elif self.config.reward_type == "follow_waypoints":
            # Reward based on minimizing distance to time-aligned waypoints plus penalty for collision/off-road
            self.base_rewards = (
                goal_achieved_weight * goal_achieved
                + collision_weight * collided
                + off_road_weight * off_road
            )

            # Extract waypoints (ground truth) at time t
            step_in_world = self.step_in_world[:, 0, :].squeeze(-1)
            batch_indices = torch.arange(step_in_world.shape[0])
            gt_agent_pos = self.log_trajectory.pos_xy[
                batch_indices, :, step_in_world, :
            ]

            gt_agent_speed = self.log_trajectory.ref_speed[
                batch_indices, :, step_in_world
            ]
            valid_mask = (
                self.log_trajectory.valids[batch_indices, :, step_in_world]
                .squeeze(-1)
                .bool()
            )

            # Get actual agent positions
            agent_state = GlobalEgoState.from_tensor(
                self.sim.absolute_self_observation_tensor(),
                self.backend,
                self.device,
            )

            actual_agent_speed = self.sim.self_observation_tensor().to_torch()[
                :, :, 0
            ]

            actual_agent_pos = torch.stack(
                [agent_state.pos_x, agent_state.pos_y], dim=-1
            )

            speed_error = (gt_agent_speed - actual_agent_speed) ** 2

            # Compute euclidean distance between agent and waypoints
            dist_to_waypoints = torch.norm(
                gt_agent_pos - actual_agent_pos, dim=-1
            )

            # Penalty for jerky movements
            if hasattr(self, "action_diff"):
                acceleration_jerk = (
                    self.action_diff[:, :, 0] ** 2
                )  # First action component is acceleration
                steering_jerk = (
                    self.action_diff[:, :, 1] ** 2
                )  # Second action component is steering

                self.smoothness_penalty = -(
                    self.config.jerk_smoothness_scale * acceleration_jerk
                    + self.config.jerk_smoothness_scale * steering_jerk
                )
            else:
                self.smoothness_penalty = torch.zeros_like(self.base_rewards)

            self.distance_penalty = (
                -self.config.waypoint_distance_scale
                * torch.log(dist_to_waypoints + 1.0)
                - self.config.speed_distance_scale
                * torch.log(speed_error + 1.0)
            )

            # Zero-out distance penalty for invalid time steps, that is,
            # The reference positions have not been observed at every time step
            # if not observed, we set the distance penalty to 0
            self.distance_penalty[~valid_mask] = 0.0

            self.distance_penalty += self.smoothness_penalty

            # Apply waypoint mask only if sampling interval is greater than 1
            if self.config.waypoint_sample_interval > 1:
                waypoint_mask = (
                    (step_in_world % self.config.waypoint_sample_interval == 0)
                    .float()
                    .unsqueeze(1)
                )
                self.distance_penalty = self.distance_penalty * waypoint_mask

            # Combine base rewards with distance penalty
            rewards = self.base_rewards + self.distance_penalty

            return rewards

    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

        # Update time in worlds
        self.step_in_world = (
            self.episode_len - self.sim.steps_remaining_tensor().to_torch()
        )

        not_done_worlds = ~self.get_dones().any(
            dim=1
        )  # Check if any agent in world is done
        self.world_time_steps[not_done_worlds] += 1

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

        if self.config.dynamics_model == "state" and self.previous_action_value_tensor.shape != self.action_value_tensor.shape:
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
            ego_state.is_collided.unsqueeze(-1),
        ]

        if mask is None:

            base_fields.append(
                self.previous_action_value_tensor[:, :, :2]
                / constants.MAX_ACTION_VALUE,  # Previous accel, steering
            )

            if self.config.add_reference_speed:

                avg_ref_speed = (
                    self.log_trajectory.clone().ref_speed.mean(axis=-1)
                    / constants.MAX_SPEED
                )

                base_fields.append(avg_ref_speed.unsqueeze(-1))

            if self.config.add_reference_path:

                state = (
                    self.sim.absolute_self_observation_tensor()
                    .to_torch()
                    .clone().to(self.device)
                )
                global_ego_pos_xy = state[:, :, :2]
                global_ego_yaw = state[:, :, 7]
                glob_reference_xy = self.log_trajectory.pos_xy
                agent_indices = torch.arange(self.max_cont_agents)
                local_reference_xy = torch.empty_like(glob_reference_xy)
                valid_timesteps_mask = self.log_trajectory.valids.bool()

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
                            ego_pos=global_ego_pos_xy[world_idx, agent_idx],
                            ego_yaw=global_ego_yaw[world_idx, agent_idx],
                            device=self.device,
                        )

                local_ref_xy_orig = local_reference_xy.clone()

                # Normalize
                local_reference_xy /= constants.MAX_REF_POINT

                # Set invalid steps to -1.0
                local_reference_xy[
                    ~valid_timesteps_mask.expand_as(local_reference_xy)
                ] = constants.INVALID_ID

                # Provide agent with index to pay attention to through one-hot encoding
                next_step_in_world = torch.clamp(
                    self.step_in_world[:, 0, :].squeeze(-1) + 1,
                    min=0,
                    max=self.episode_len,
                )
                time_one_hot = torch.zeros(
                    (
                        self.num_worlds,
                        self.max_agent_count,
                        self.reference_path_length,
                        1,
                    ),
                    device=self.device,
                )
                time_one_hot[:, :, next_step_in_world, :] = 1.0

                # Make unnormalized reference path available for plotting
                self.reference_path = torch.cat(
                    (local_ref_xy_orig, time_one_hot), dim=-1
                )

                reference_path = torch.cat(
                    (local_reference_xy, time_one_hot), dim=-1
                )

                # Flatten the dimensions for stacking
                base_fields.append(reference_path.flatten(start_dim=2))

                # batch_size = local_reference_xy.shape[0]
                # num_points = local_reference_xy.shape[1]
                # time_steps = local_reference_xy.shape[2]

                # Create dropout mask for the time dimension
                # Shape: [batch_size, num_points, time_steps, 1]
                # point_dropout_mask = torch.bernoulli(
                #     torch.ones(
                #         batch_size,
                #         num_points,
                #         time_steps,
                #         1,
                #         device=local_reference_xy.device,
                #     )
                #     * (1 - self.config.prob_reference_dropout)
                # ).bool()

                # Apply dropout mask
                # self.local_reference_xy = (
                #     local_reference_xy * point_dropout_mask
                # )

            if self.config.reward_type == "reward_conditioned":

                # Create expanded weights for all environments
                # Expand from [max_agents, 3] to [num_worlds, max_agents]
                collision_weights = self.reward_weights_tensor[:, 0].expand(
                    self.num_worlds, -1
                )
                goal_weights = self.reward_weights_tensor[:, 1].expand(
                    self.num_worlds, -1
                )
                off_road_weights = self.reward_weights_tensor[:, 2].expand(
                    self.num_worlds, -1
                )

                full_fields = base_fields + [
                    collision_weights,
                    goal_weights,
                    off_road_weights,
                ]
                return torch.stack(full_fields).permute(1, 2, 0)
            else:
                return torch.cat(base_fields, dim=-1)
        else:

            base_fields.append(
                self.previous_action_value_tensor[mask][:, :2]
                / constants.MAX_ACTION_VALUE,  # Previous accel, steering
            )

            if self.config.add_reference_speed:
                avg_ref_speed = (
                    self.log_trajectory.ref_speed[mask].clone().mean(axis=-1)
                    / constants.MAX_SPEED
                )
                base_fields.append(avg_ref_speed.unsqueeze(-1))

            if self.config.add_reference_path:

                # State information
                state = (
                    self.sim.absolute_self_observation_tensor()
                    .to_torch()
                    .clone()[mask]
                ).to(self.device)
                global_ego_pos_xy = state[:, :2]  # Shape: [batch, 2]
                global_ego_yaw = state[:, 7]  # Shape: [batch]
                global_reference_xy = self.log_trajectory.pos_xy.clone()[mask]
                valid_timesteps_mask = self.log_trajectory.valids.bool()[mask]
                batch_size = global_reference_xy.shape[0]
                batch_indices = torch.arange(batch_size)

                # Translate all points to a local coordinate frame
                translated = global_reference_xy - global_ego_pos_xy.unsqueeze(
                    1
                )

                # Create rotation matrices for all agents at once
                cos_yaw = torch.cos(global_ego_yaw)
                sin_yaw = torch.sin(global_ego_yaw)

                # Create batch of rotation matrices: [batch, 2, 2]
                rotation_matrices = torch.stack(
                    [
                        torch.stack([cos_yaw, sin_yaw], dim=1),
                        torch.stack([-sin_yaw, cos_yaw], dim=1),
                    ],
                    dim=1,
                )  # Shape: [batch, 2, 2]

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

                # Provide agent with index to pay attention to through one-hot encoding
                next_step_in_world = torch.clamp(
                    self.step_in_world[mask] + 1, min=0, max=self.episode_len
                )
                time_one_hot = torch.zeros(
                    (batch_size, self.reference_path_length, 1),
                    device=self.device,
                )
                time_one_hot[batch_indices, next_step_in_world] = 1.0

                # Stack
                reference_path = torch.cat(
                    (local_reference_xy, time_one_hot), dim=2
                )

                self.reference_path = torch.cat(
                    (local_reference_xy_orig, time_one_hot), dim=2
                )

                # Stack
                base_fields.append(reference_path.flatten(start_dim=1))

            if self.config.reward_type == "reward_conditioned":
                # For masked agents, we need to extract agent indices from the mask
                world_indices, agent_indices = torch.where(mask)

                # Get the reward weights for these specific agents
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

    def _get_vbd_obs(self, mask=None):
        """
        Get ego-centric VBD trajectory observations for controlled agents.

        Args:
            mask: Optional mask to filter agents

        Returns:
            Tensor of ego-centric VBD trajectories
        """
        if not self.use_vbd or self.vbd_trajectories is None:
            return torch.Tensor().to(self.device)

        # Get current agent positions and orientations
        agent_state = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=self.sim.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # Initialize output tensor
        traj_feature_dim = (
            self.vbd_trajectories.shape[2] * self.vbd_trajectories.shape[3]
        )

        if mask is not None:
            # Count valid agents for output tensor size
            valid_count = mask.sum().item()
            ego_vbd_trajectories = torch.zeros(
                (valid_count, traj_feature_dim), device=self.device
            )

            # Track which output index we're filling
            out_idx = 0

            # Process each world
            for w in range(self.num_worlds):
                # Get valid agent indices for this world
                world_mask = mask[w]
                agent_indices = torch.where(world_mask)[0]

                if len(agent_indices) == 0:
                    continue

                # Extract ego positions and yaws for these agents
                ego_pos_x = agent_state.pos_x[w, agent_indices]
                ego_pos_y = agent_state.pos_y[w, agent_indices]
                ego_yaw = agent_state.rotation_angle[w, agent_indices]

                # Process each agent in this world
                for i, agent_idx in enumerate(agent_indices):
                    # Get global trajectory for this agent
                    global_traj = self.vbd_trajectories[w, agent_idx]

                    # Create 2D rotation matrix for this agent
                    cos_yaw = torch.cos(ego_yaw[i])
                    sin_yaw = torch.sin(ego_yaw[i])
                    rotation_matrix = torch.tensor(
                        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
                        device=self.device,
                    )

                    # Transform positions using matrix multiplication
                    pos_xy = global_traj[:, :2]
                    ego_pos = torch.tensor(
                        [ego_pos_x[i], ego_pos_y[i]], device=self.device
                    ).reshape(1, 2)
                    translated_pos = (
                        pos_xy - ego_pos
                    )  # Broadcasting to subtract from all timesteps
                    rotated_pos = torch.matmul(
                        translated_pos, rotation_matrix.T
                    )

                    # Transform velocities (only rotation, no translation)
                    vel_xy = global_traj[:, 3:5]
                    rotated_vel = torch.matmul(vel_xy, rotation_matrix.T)

                    # Create transformed trajectory
                    transformed_traj = torch.zeros_like(global_traj)
                    transformed_traj[:, :2] = rotated_pos
                    transformed_traj[:, 2] = (
                        global_traj[:, 2] - ego_yaw[i]
                    )  # Adjust heading
                    transformed_traj[:, 3:5] = rotated_vel

                    # Flatten and add to output
                    ego_vbd_trajectories[out_idx] = transformed_traj.reshape(
                        -1
                    )
                    out_idx += 1

            if self.config.norm_obs:
                ego_vbd_trajectories = self._normalize_vbd_obs(
                    ego_vbd_trajectories, self.vbd_trajectories.shape[2]
                )

            return ego_vbd_trajectories

        else:
            # Without mask, process all agents in all worlds
            ego_vbd_trajectories = torch.zeros(
                (self.num_worlds, self.max_agent_count, traj_feature_dim),
                device=self.device,
            )

            # Process each world
            for w in range(self.num_worlds):
                # Get controlled agent indices for this world
                valid_mask = self.cont_agent_mask[w]
                world_agent_indices = torch.where(valid_mask)[0]

                if len(world_agent_indices) == 0:
                    continue

                # Extract ego positions and yaws
                ego_pos_x = agent_state.pos_x[w]
                ego_pos_y = agent_state.pos_y[w]
                ego_yaw = agent_state.rotation_angle[w]

                # Process each agent in this world
                for agent_idx in world_agent_indices:
                    # Get global trajectory
                    global_traj = self.vbd_trajectories[w, agent_idx]

                    # Create 2D rotation matrix for this agent
                    cos_yaw = torch.cos(ego_yaw[agent_idx])
                    sin_yaw = torch.sin(ego_yaw[agent_idx])
                    rotation_matrix = torch.tensor(
                        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
                        device=self.device,
                    )

                    # Transform positions
                    pos_xy = global_traj[:, :2]
                    ego_pos = torch.tensor(
                        [ego_pos_x[agent_idx], ego_pos_y[agent_idx]],
                        device=self.device,
                    ).reshape(1, 2)
                    translated_pos = pos_xy - ego_pos
                    rotated_pos = torch.matmul(
                        translated_pos, rotation_matrix.T
                    )

                    # Transform velocities
                    vel_xy = global_traj[:, 3:5]
                    rotated_vel = torch.matmul(vel_xy, rotation_matrix.T)

                    # Create transformed trajectory
                    transformed_traj = torch.zeros_like(global_traj)
                    transformed_traj[:, :2] = rotated_pos
                    transformed_traj[:, 2] = (
                        global_traj[:, 2] - ego_yaw[agent_idx]
                    )
                    transformed_traj[:, 3:5] = rotated_vel

                    # Flatten and add to output
                    ego_vbd_trajectories[
                        w, agent_idx
                    ] = transformed_traj.reshape(-1)

            if self.config.norm_obs:
                ego_vbd_trajectories = self._normalize_vbd_obs(
                    ego_vbd_trajectories, self.vbd_trajectories.shape[2]
                )

            return ego_vbd_trajectories

    def _normalize_vbd_obs(self, trajectories_flat, traj_len):
        """
        Normalize flattened VBD trajectory values to be between -1 and 1, with clipping.

        Args:
            trajectories_flat: Flattened tensor containing trajectory data
            traj_len: Number of trajectory steps

        Returns:
            Normalized flattened trajectories tensor
        """
        # Get original shape for proper reshaping
        original_shape = trajectories_flat.shape

        # Calculate feature dimension
        feature_dim = 5  # x, y, yaw, vel_x, vel_y

        # Reshape to separate the features
        if len(original_shape) == 2:  # (num_agents, flattened_features)
            traj_features = trajectories_flat.reshape(
                -1, traj_len, feature_dim
            )
        else:  # (num_worlds, max_agents, flattened_features)
            traj_features = trajectories_flat.reshape(
                original_shape[0], original_shape[1], traj_len, feature_dim
            )

        # Normalize each feature
        # x, y positions
        traj_features[..., 0] = normalize_min_max(
            tensor=traj_features[..., 0],
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )
        traj_features[..., 1] = normalize_min_max(
            tensor=traj_features[..., 1],
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )

        # Normalize yaw angle
        traj_features[..., 2] = (
            traj_features[..., 2] / constants.MAX_ORIENTATION_RAD
        )

        # Normalize velocities
        traj_features[..., 3] = traj_features[..., 3] / constants.MAX_SPEED
        traj_features[..., 4] = traj_features[..., 4] / constants.MAX_SPEED

        # Clip all values to the [-1, 1] range
        traj_features = torch.clamp(traj_features, min=-1.0, max=1.0)

        # Reshape back to original format
        return traj_features.reshape(original_shape)

    def get_obs(self, mask=None):
        """Get observation: Combine different types of environment information into a single tensor.
        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """
        # Base observations
        ego_states = self._get_ego_state(mask)
        partner_observations = self._get_partner_obs(mask)
        road_map_observations = self._get_road_map_obs(mask)

        if self.use_vbd and self.config.vbd_in_obs:
            # Add ego-centric VBD trajectories
            vbd_observations = self._get_vbd_obs(mask)

            obs = torch.cat(
                (
                    ego_states,
                    partner_observations,
                    road_map_observations,
                    vbd_observations,
                ),
                dim=-1,
            )
        else:
            obs = torch.cat(
                (
                    ego_states,
                    partner_observations,
                    road_map_observations,
                ),
                dim=-1,
            )

        return obs

    def get_controlled_agents_mask(self):
        """Get the control mask. Shape: [num_worlds, max_agent_count]"""
        return (
            self.sim.controlled_state_tensor().to_torch().clone() == 1
        ).squeeze(axis=2)

    def advance_sim_with_log_playback(self, init_steps=0):
        """Advances the simulator by stepping the objects with the logged human trajectories.

        Args:
            init_steps (int): Number of warmup steps.
        """
        if init_steps >= self.config.episode_len:
            raise ValueError(
                "The length of the expert trajectory is 91,"
                f"so init_steps = {init_steps} should be < than 91."
            )

        self.init_frames = []

        self.log_playback_traj, _, _, _ = self.get_expert_actions()

        for time_step in range(init_steps):
            self.step_dynamics(
                actions=self.log_playback_traj[:, :, time_step, :]
            )

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
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)

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

        # Load VBD trajectories for the new batch if VBD is enabled
        if self.use_vbd:
            self._load_vbd_trajectories()

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)

        # Obtain new log trajectory
        self.log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
            device=self.device
        )

    def _load_vbd_trajectories(self):
        """Load VBD trajectories directly from the simulator."""
        if not self.use_vbd:
            return

        # Get VBD trajectories from the simulator
        vbd_traj = VBDTrajectory.from_tensor(
            self.sim.vbd_trajectory_tensor(),
            backend=self.backend,
            device=self.device,
        )

        means_xy = (
            self.sim.world_means_tensor().to_torch()[:, :2].to(self.device)
        )
        vbd_traj.restore_mean(mean_x=means_xy[:, 0], mean_y=means_xy[:, 1])

        self.vbd_trajectories = vbd_traj.trajectories

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
            device=self.device
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


if __name__ == "__main__":

    env_config = EnvConfig(
        dynamics_model="state",
        reward_type="follow_waypoints",
        add_reference_path=True,
        init_mode="womd_tracks_to_predict"
    )
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/temp",
        batch_size=1,
        dataset_size=1,
        sample_with_replacement=False,
        shuffle=False,
        file_prefix=""
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=64,  # Number of agents to control
        device="cpu",
    )

    control_mask = env.cont_agent_mask
    print(f"Number of controlled agents: {control_mask.sum()}")

    # Rollout
    obs = env.reset(mask=control_mask)

    sim_frames = []
    agent_obs_frames = []

    expert_actions, _, _, _ = env.get_expert_actions()

    env_idx = 0
    idx_to_id = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        device=env.device,
    ).id
    
    highlight_agent = torch.where(idx_to_id[env_idx] == 2293)[0].item()

    agent_positions = []
    agent_positions.append(GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        device=env.device,
    ).pos_xy[env_idx, highlight_agent])

    print(f"Highlighted agent: {highlight_agent}")
    print(f"Position: {agent_positions[-1]}")

    for t in range(90):
        print(f"Step: {t+1}")

        # Step the environment
        expert_actions, _, _, _ = env.get_expert_actions()
        env.step_dynamics(expert_actions[:, :, t - 1, :])

        agent_positions.append(GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            device=env.device,
        ).pos_xy[env_idx, highlight_agent])

        # Make video
        sim_states = env.vis.plot_simulator_state(
            env_indices=[env_idx],
            zoom_radius=50,
            time_steps=[t],
            center_agent_indices=[highlight_agent],
            plot_waypoints=True,
        )

        agent_obs = env.vis.plot_agent_observation(
            env_idx=env_idx,
            agent_idx=highlight_agent,
            figsize=(10, 10),
            trajectory=env.reference_path[highlight_agent, :, :].to(
                env.device
            ),
        )

        sim_frames.append(img_from_fig(sim_states[0]))
        agent_obs_frames.append(img_from_fig(agent_obs))

        world_time_steps = (
            torch.Tensor([t]).repeat((1, env.num_worlds)).long().to(env.device)
        )

        obs = env.get_obs(control_mask)
        reward = env.get_rewards()

        print(f"A_t: {expert_actions[env_idx, highlight_agent, t, :]}")
        print(f"R_t+1: {reward[env_idx, highlight_agent]}")
        print(f"Position: {agent_positions[-1]}")

        done = env.get_dones()
        info = env.get_infos()

        print(done[env.cont_agent_mask])

        # if done.all().bool():
        #     # Check resetting behavior
        #     _ = env.reset(control_mask)
        #     env.step_dynamics(expert_actions[:, :, 0, :])

    env.close()

    media.write_video(
        "sim_video.gif", np.array(sim_frames), fps=10, codec="gif"
    )
    media.write_video(
        f"obs_video_env_{env_idx}_agent_{highlight_agent}.gif",
        np.array(agent_obs_frames),
        fps=10,
        codec="gif",
    )
