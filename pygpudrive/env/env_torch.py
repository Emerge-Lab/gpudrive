"""Torch Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
import gpudrive
import imageio
from itertools import product

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.base_env import GPUDriveGymEnv

from pygpudrive.datatypes.observation import (
    LocalEgoState,
    PartnerObs,
    LidarObs,
)
from pygpudrive.datatypes.trajectory import LogTrajectory
from pygpudrive.datatypes.roadgraph import LocalRoadGraphPoints

from pygpudrive.visualize.core import MatplotlibVisualizer


class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        scene_config,
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
        backend="torch",
    ):
        # Initialization of environment configurations
        self.config = config
        self.scene_config = scene_config
        self.num_worlds = scene_config.num_scenes
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config
        self.backend = backend
        self.max_num_agents_in_scene = self.config.max_num_agents_in_scene

        # Environment parameter setup
        params = self._setup_environment_parameters()

        # Initialize simulator with parameters
        self.sim = self._initialize_simulator(params, scene_config)

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.get_obs().shape[-1],)
        )
        self._setup_action_space(action_type)
        self.info_dim = 5  # Number of info features
        self.episode_len = self.config.episode_len

        # Rendering setup
        self.visualizer = self._setup_rendering()

    def reset(self):
        """Reset the worlds and return the initial observations."""
        self.sim.reset(list(range(self.num_worlds)))

        # If there are warmup steps (by default, init_steps=0),
        # advance the simulator before returning the first observation
        self.warmup_trajectory = self._update_sim_state_by_log(
            init_steps=self.config.init_steps,
            render_init=self.render_config.render_init,
        )

        return self.get_obs()

    def get_dones(self):
        return self.sim.done_tensor().to_torch().squeeze(dim=2).to(torch.float)

    def get_infos(self):
        return (
            self.sim.info_tensor()
            .to_torch()
            .squeeze(dim=2)
            .to(torch.float)
            .to(self.device)
        )

    def get_rewards(
        self, collision_weight=0, goal_achieved_weight=1.0, off_road_weight=0
    ):
        """Obtain the rewards for the current step.
        By default, the reward is a weighted combination of the following components:
        - collision
        - goal_achieved
        - off_road

        The importance of each component is determined by the weights.
        """
        if self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().squeeze(dim=2)

        elif self.config.reward_type == "weighted_combination":
            # Return the weighted combination of the reward components
            info_tensor = self.sim.info_tensor().to_torch()
            off_road = info_tensor[:, :, 0].to(torch.float)

            # True if the vehicle collided with another road object
            # (i.e. a cyclist or pedestrian)
            collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
            goal_achieved = info_tensor[:, :, 3].to(torch.float)

            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            return weighted_rewards

    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

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
                action_value_tensor = self.action_keys_tensor[actions]

            elif actions.dim() == 3:
                if actions.shape[2] == 1:
                    actions = actions.squeeze(dim=2).to(self.device)
                    action_value_tensor = self.action_keys_tensor[actions]
                elif (
                    actions.shape[2] == 3
                ):  # Assuming we are given the actual action values
                    # (acceleration, steering, heading)
                    action_value_tensor = actions.to(self.device)
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")

        else:
            action_value_tensor = actions.to(self.device)

        # Feed the action values to gpudrive
        self._copy_actions_to_simulator(action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provived actions to the simulator."""
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
        else:
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_1 = self.steer_actions.clone().cpu().numpy()
            action_2 = self.accel_actions.clone().cpu().numpy()
            action_3 = self.head_actions.clone().cpu().numpy()

        action_space = Tuple(
            (
                Box(action_1.min(), action_1.max(), shape=(1,)),
                Box(action_2.min(), action_2.max(), shape=(1,)),
                Box(action_3.min(), action_3.max(), shape=(1,)),
            )
        )
        return action_space

    def get_obs(self):
        """Get observation: Combine different types of environment information
        into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        # EGO STATE
        if self.config.ego_state:
            ego_state = LocalEgoState.from_tensor(
                self_obs_tensor=self.sim.self_observation_tensor(),
                backend=self.backend,
            )
            if self.config.norm_obs:
                ego_state.normalize()

            return (
                torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                    ]
                )
                .permute(1, 2, 0)
                .to(self.device)
            )
        else:
            return torch.Tensor().to(self.device)

    def _get_partner_obs(self):
        """Get partner observations."""
        if self.config.partner_obs:
            partner_obs = PartnerObs.from_tensor(
                partner_obs_tensor=self.sim.partner_observations_tensor(),
                backend=self.backend,
            )

            if self.config.norm_obs:
                partner_obs.normalize()
                partner_obs.one_hot_encode_agent_types()

            return (
                torch.concat(
                    [
                        partner_obs.speed,
                        partner_obs.rel_pos_x,
                        partner_obs.rel_pos_y,
                        partner_obs.orientation,
                        partner_obs.vehicle_length,
                        partner_obs.vehicle_width,
                        # partner_obs.agent_type,
                    ],
                    dim=-1,
                )
                .flatten(start_dim=2)
                .to(self.device)
            )

        else:
            return torch.Tensor().to(self.device)

    def _get_road_map_obs(self):
        """Get road map observations."""
        if self.config.road_map_obs:
            roadgraph = LocalRoadGraphPoints.from_tensor(
                local_roadgraph_tensor=self.sim.agent_roadmap_tensor(),
                backend=self.backend,
            )

            if self.config.norm_obs:
                roadgraph.normalize()
                roadgraph.one_hot_encode_road_point_types()

            return (
                torch.cat(
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
                )
                .flatten(start_dim=2)
                .to(self.device)
            )

        else:
            return torch.Tensor().to(self.device)

    def _get_lidar_obs(self):
        """Get lidar observations."""
        if self.config.lidar_obs:
            lidar = LidarObs.from_tensor(
                lidar_tensor=self.sim.lidar_tensor(),
                backend=self.backend,
            )

            return (
                torch.cat(
                    [
                        lidar.agent_samples,
                        lidar.road_edge_samples,
                        lidar.road_line_samples,
                    ],
                    dim=-1,
                )
                .flatten(start_dim=2)
                .to(self.device)
            )
        else:
            return torch.Tensor().to(self.device)

    def get_obs(self):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        ego_states = self._get_ego_state()

        partner_observations = self._get_partner_obs()

        road_map_observations = self._get_road_map_obs()

        lidar_obs = self._get_lidar_obs()

        obs_filtered = torch.cat(
            (
                ego_states,
                partner_observations,
                road_map_observations,
                lidar_obs,
            ),
            dim=-1,
        )

        return obs_filtered

    def _update_sim_state_by_log(self, init_steps=0, render_init=False):
        """Advances the simulator by stepping the objects with the inferred human actions.

        Args:
            init_steps (int): Number of warmup steps to perform.
        """

        if init_steps >= self.config.episode_len:
            raise ValueError(
                "The length of the expert trajectory is 91,"
                "so num_steps should be less than 91."
            )

        self.inferred_playback_actions = self.get_expert_actions()

        if self.config.enable_vbd:
            from vbd.data.data_utils import calculate_relations

            # Storage
            agents_history = torch.zeros(
                (self.num_worlds, self.max_cont_agents, init_steps + 1, 8)
            )  # (32, 11, 8)
            agents_future = torch.zeros(
                (
                    self.num_worlds,
                    self.max_cont_agents,
                    self.config.episode_len - (init_steps + 1),
                    5,
                )
            )
            # Zeroth step
            agents_history[:, :, 0, :], _, _ = self.construct_agent_traj()

        self.init_frames = []
        for time_step in range(init_steps):
            self.step_dynamics(
                actions=self.inferred_playback_actions[:, :, time_step, :]
            )

            if render_init:  # Render the initial frames
                self.init_frames.append(self.render())

            (
                agents_history[:, :, time_step + 1, :],
                _,
                _,
            ) = self.construct_agent_traj()

        if self.config.enable_vbd:
            # Get the agent trajectories
            _, agents_type, agents_interested = self.construct_agent_traj()

            # Global polylines tensor: Shape (256, 30, 5)
            polylines, polylines_valid = self.construct_polylines()

            # Empty (16, 3)
            traffic_light_points = torch.zeros((1, 16, 3))

            # Controlled agents
            agents_id = torch.nonzero(self.cont_agent_mask[0, :]).permute(1, 0)

            # Compute relations at the end
            relations = calculate_relations(
                agents_history.squeeze(0),
                polylines.squeeze(0),
                traffic_light_points.squeeze(0),
            )

            data_dict = {
                "agents_history": agents_history,
                "agents_interested": agents_interested,
                "agents_type": agents_type.long(),
                "agents_future": agents_future,
                "traffic_light_points": traffic_light_points,
                "polylines": polylines,
                "polylines_valid": polylines_valid,
                "relations": torch.Tensor(relations).unsqueeze(0),
                "agents_id": agents_id,
                "anchors": torch.zeros((1, 32, 64, 2)),  # Placeholder
            }

            return data_dict
        else:
            return None

    def construct_polylines(self):
        """Get the global polylines information."""

        # Features: p_x, p_y, heading, traffic_light_state, lane_type
        global_roadmap = self.sim.agent_roadmap_tensor().to_torch()

        num_road_points = global_roadmap.shape[2]

        polylines = torch.cat(
            [
                global_roadmap[:, :, :, :2],  # x, y (3D tensor)
                global_roadmap[:, :, :, 5:6],  # heading (unsqueezed to 3D)
                torch.zeros_like(
                    global_roadmap[:, :, :, 5:6]
                ),  # traffic_light_state (unsqueezed to 3D)
                global_roadmap[
                    :, :, :, 6:7
                ].long(),  # lane_type (unsqueezed to 3D)
            ],
            dim=-1,  # Concatenate along the last dimension
        )

        # Throw out garbage values
        condition = (polylines[:, :, :, 4] < 0) | (polylines[:, :, :, 4] > 6)

        polylines[:, :, :, 4] = torch.where(
            condition,
            torch.tensor(0, dtype=polylines.dtype),
            polylines[:, :, :, 4],
        )

        # TODO(dc): Map lane type to what vbd expects (ie what is used in waymax)
        # ...

        # TODO(dc): Find out 30 shape
        polylines = polylines[:, :30, :]

        # Condition to check if the value is an integer between 0 and 6
        condition = (
            (polylines[:, :, :, 4] >= 0)
            & (polylines[:, :, :, 4] <= 6)
            & (polylines[:, :, :, 4] == polylines[:, :, :, 4].long().float())
        )

        # TODO(dc): Create the new tensor, setting 1 where the condition is true, and 0 otherwise
        polylines_valid = torch.ones((self.num_worlds, num_road_points))

        return polylines.permute(0, 2, 1, 3), polylines_valid

    def construct_agent_traj(self):
        """Get the agent trajectory information."""
        global_traj = self.sim.absolute_self_observation_tensor().to_torch()
        global_traj[~self.cont_agent_mask] = 0.0
        global_traj = global_traj[:, : self.max_cont_agents, :]

        # x, y, heading, vel_x, vel_y, len, width, height
        agents_history = torch.cat(
            [
                global_traj[:, :, :2],  # x, y
                global_traj[:, :, 7:8],  # TODO(dc): yaw (placeholder)
                torch.zeros_like(
                    global_traj[:, :, :2]
                ),  # velocity xy (placeholder)
                global_traj[:, :, 10:12],  # vehicle length, width
                torch.zeros_like(
                    global_traj[:, :, 0:1]
                ),  # TODO(dc): height (placeholder)
            ],
            dim=-1,
        )
        # Currently, all agents are vehicles, encoding as type 1
        agents_type = torch.zeros([self.num_worlds, self.max_cont_agents])
        agents_type[self.cont_agent_mask] = 1

        # 10 if we are controlling the agent, 1 otherwise
        agents_interested = torch.ones([self.num_worlds, self.max_cont_agents])
        agents_interested[self.cont_agent_mask] = 10

        return agents_history, agents_type, agents_interested

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (self.sim.controlled_state_tensor().to_torch() == 1).squeeze(
            axis=2
        )

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= constants.MAX_SPEED
        state[:, :, 1] /= constants.MAX_VEH_LEN
        state[:, :, 2] /= constants.MAX_VEH_WIDTH

        # Relative goal coordinates
        state[:, :, 3] = self.normalize_tensor(
            state[:, :, 3],
            constants.MIN_REL_GOAL_COORD,
            constants.MAX_REL_GOAL_COORD,
        )
        state[:, :, 4] = self.normalize_tensor(
            state[:, :, 4],
            # do the same
            constants.MIN_REL_GOAL_COORD,
            constants.MAX_REL_GOAL_COORD,
        )

        # Uncommment this to exclude the collision state
        # (1 if vehicle is in collision, 1 otherwise)
        # state = state[:, :, :5]

        return state

    def get_expert_actions(self, debug_world_idx=None, debug_veh_idx=None):
        """Get expert actions for the full trajectories across worlds."""

        expert_traj = self.sim.expert_trajectory_tensor().to_torch()

        # Global positions
        positions = expert_traj[:, :, : 2 * self.episode_len].view(
            self.num_worlds, self.max_agent_count, self.episode_len, -1
        )

        # Global velocity
        velocity = expert_traj[
            :, :, 2 * self.episode_len : 4 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        headings = expert_traj[
            :, :, 4 * self.episode_len : 5 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        inferred_expert_actions = expert_traj[
            :, :, 6 * self.episode_len : 16 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        if self.config.dynamics_model == "delta_local":
            inferred_actions = log_trajectory[
                :, :, -3 * constants.LOG_TRAJECTORY_LEN :
            ].view(
                self.num_worlds,
                self.max_agent_count,
                constants.LOG_TRAJECTORY_LEN,
                -1,
            )
        )

        if self.config.dynamics_model == "delta_local":
            inferred_expert_actions = expert_traj[
                :, :, -3 * self.episode_len :
            ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)
            inferred_expert_actions[..., 0] = torch.clamp(
                inferred_expert_actions[..., 0], -6, 6
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

        else:  # classic or bicycle
            inferred_expert_actions = inferred_expert_actions[..., :3]
            inferred_expert_actions[..., 0] = torch.clamp(
                inferred_expert_actions[..., 0], -6, 6
            )
            inferred_expert_actions[..., 1] = torch.clamp(
                inferred_expert_actions[..., 1], -0.3, 0.3
            )
        velo2speed = None
        debug_positions = None
        if debug_world_idx is not None and debug_veh_idx is not None:
            velo2speed = (
                torch.norm(velocity[debug_world_idx, debug_veh_idx], dim=-1)
                / self.config.max_speed
            )
            positions[..., 0] = self.normalize_tensor(
                positions[..., 0],
                self.config.min_rel_goal_coord,
                self.config.max_rel_goal_coord,
            )
            positions[..., 1] = self.normalize_tensor(
                positions[..., 1],
                self.config.min_rel_goal_coord,
                self.config.max_rel_goal_coord,
            )
            debug_positions = positions[debug_world_idx, debug_veh_idx]
        return inferred_expert_actions, velo2speed, debug_positions

    def normalize_and_flatten_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: torch.Tensor of shape (
                num_worlds,
                kMaxAgentCount,
                kMaxAgentCount - 1,
                num_features
            )
        """

        # TODO: Fix (there should not be nans in the obs)
        obs = torch.nan_to_num(obs, nan=0)

        # Speed
        obs[:, :, :, 0] /= constants.MAX_SPEED

        # Relative position
        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )
        obs[:, :, :, 2] = self.normalize_tensor(
            obs[:, :, :, 2],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )

        # Orientation (heading)
        obs[:, :, :, 3] /= constants.MAX_ORIENTATION_RAD

        # Vehicle length and width
        obs[:, :, :, 4] /= constants.MAX_VEH_LEN
        obs[:, :, :, 5] /= constants.MAX_VEH_WIDTH

        # One-hot encode the type of the other visible objects
        one_hot_encoded_object_types = self.one_hot_encode_object_type(
            obs[:, :, :, 6]
        )

        # Concat the one-hot encoding with the rest of the features
        obs = torch.concat(
            (obs[:, :, :, :6], one_hot_encoded_object_types), dim=-1
        )

        return obs.flatten(start_dim=2)

    def one_hot_encode_roadpoints(self, roadmap_type_tensor):

        # Set garbage object types to zero
        road_types = torch.where(
            (roadmap_type_tensor < self.MIN_OBJ_ENTITY_ENUM)
            | (roadmap_type_tensor > self.ROAD_MAP_OBJECT_TYPES),
            0.0,
            roadmap_type_tensor,
        ).int()

        return torch.nn.functional.one_hot(
            road_types.long(),
            num_classes=self.ROAD_MAP_OBJECT_TYPES,
        )

    def one_hot_encode_object_type(self, object_type_tensor):
        """One-hot encode the object type."""

        VEHICLE = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Vehicle]
        PEDESTRIAN = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Pedestrian]
        CYCLIST = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Cyclist]
        PADDING = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType._None]

        # Set garbage object elements to zero
        object_types = torch.where(
            (object_type_tensor < self.MIN_OBJ_ENTITY_ENUM)
            | (object_type_tensor > self.MAX_OBJ_ENTITY_ENUM),
            0.0,
            object_type_tensor,
        ).int()

        one_hot_object_type = torch.nn.functional.one_hot(
            torch.where(
                condition=(object_types == VEHICLE)
                | (object_types == PEDESTRIAN)
                | (object_types == CYCLIST)
                | object_types
                == PADDING,
                input=object_types,
                other=0,
            ).long(),
            num_classes=self.ROAD_OBJECT_TYPES,
        )
        return one_hot_object_type

    @property
    def step_in_episode(self):
        return (
            self.config.episode_len
            - self.sim.steps_remaining_tensor().to_torch().flatten()[0].item()
        )

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""

        # Road point coordinates
        obs[:, :, :, 0] = self.normalize_tensor(
            obs[:, :, :, 0],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )

        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )

        # Road line segment length
        obs[:, :, :, 2] /= constants.MAX_ROAD_LINE_SEGMENT_LEN

        # Road scale (width and height)
        obs[:, :, :, 3] /= constants.MAX_ROAD_SCALE
        # obs[:, :, :, 4] seems already scaled

        # Road point orientation
        obs[:, :, :, 5] /= constants.MAX_ORIENTATION_RAD

        # Road types: one-hot encode them
        one_hot_road_types = self.one_hot_encode_roadpoints(obs[:, :, :, 6])

        # Concatenate the one-hot encoding with the rest of the features
        obs = torch.cat((obs[:, :, :, :6], one_hot_road_types), dim=-1)

        return obs.flatten(start_dim=2)


if __name__ == "__main__":

    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_NUM_OBJECTS = 32
    NUM_WORLDS = 1

    env_config = EnvConfig(
        init_steps=10, enable_vbd=True, dynamics_model="state"
    )
    render_config = RenderConfig()
    scene_config = SceneConfig("data/processed/training", NUM_WORLDS)

    # MAKE ENV
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
        device="cpu",
        render_config=render_config,
    )

    # RUN
    obs = env.reset()
    frames = []

    expert_actions, _, _, _ = env.get_expert_actions()

    for t in range(TOTAL_STEPS):
        print(f"Step: {t}")

        # Step the environment
        env.step_dynamics(expert_actions[:, :, t, :])

        frames.append(env.render())

        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

    # import imageio
    imageio.mimsave("world1.gif", np.array(frames))

    env.close()
