import torch
from dataclasses import dataclass
from gpudrive.env import constants
from gpudrive.utils.geometry import (
    normalize_min_max,
    normalize_min_max_inplace,
)
import madrona_gpudrive

AGENT_SCALE = madrona_gpudrive.vehicleScale


class LocalEgoState:
    """A class to represent the ego state of the agent in relative coordinates.
    Initialized from self_observation_tensor (src/bindings). For details, see
    `SelfObservation` in src/types.hpp.

    Attributes:
        speed: Speed of the agent in relative coordinates.
        vehicle_length: Length of the agent's bounding box.
        vehicle_width: Width of the agent's bounding box.
        vehicle_height: Height of the agent's bounding box.
        rel_goal_x: Relative x-coordinate to the goal.
        rel_goal_y: Relative y-coordinate to the goal.
        is_collided: Whether the agent is in collision with another object.
        id: Unique identifier of the agent.
    """

    def __init__(self, self_obs_tensor: torch.Tensor, mask=None):
        """Initializes the ego state with an observation tensor."""
        if mask is not None:
            self_obs_tensor = self_obs_tensor[mask]
            self.speed = self_obs_tensor[:, 0]
            self.vehicle_length = self_obs_tensor[:, 1] * AGENT_SCALE
            self.vehicle_width = self_obs_tensor[:, 2] * AGENT_SCALE
            self.vehicle_height = self_obs_tensor[:, 3]
            self.rel_goal_x = self_obs_tensor[:, 4]
            self.rel_goal_y = self_obs_tensor[:, 5]
            self.is_collided = self_obs_tensor[:, 6]
            self.id = self_obs_tensor[:, 7]
        else:
            self.speed = self_obs_tensor[:, :, 0]
            self.vehicle_length = self_obs_tensor[:, :, 1] * AGENT_SCALE
            self.vehicle_width = self_obs_tensor[:, :, 2] * AGENT_SCALE
            self.vehicle_height = self_obs_tensor[:, :, 3]
            self.rel_goal_x = self_obs_tensor[:, :, 4]
            self.rel_goal_y = self_obs_tensor[:, :, 5]
            self.is_collided = self_obs_tensor[:, :, 6]
            self.id = self_obs_tensor[:, :, 7]

    @classmethod
    def from_tensor(
        cls,
        self_obs_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
        mask=None,
    ):
        """
        Creates an LocalEgoState from the agent_observation_tensor.
        """
        if backend == "torch":
            tensor = self_obs_tensor.to_torch().clone().to(device)
            obj = cls(tensor, mask=mask)
            return obj

        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the ego state to be between -1 and 1."""
        self.speed /= constants.MAX_SPEED
        self.vehicle_length /= constants.MAX_VEH_LEN
        self.vehicle_width /= constants.MAX_VEH_WIDTH
        self.vehicle_height /= constants.MAX_VEH_HEIGHT

        self.rel_goal_x = normalize_min_max(
            tensor=self.rel_goal_x,
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )
        self.rel_goal_y = normalize_min_max(
            tensor=self.rel_goal_y,
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of the ego state tensor."""
        return self.speed.shape


class GlobalEgoState:
    """A class to represent the ego state of the agent in global coordinates.
    Initialized from absolute_self_observation_tensor (src/bindings). For details, see
    `AbsoluteSelfObservation` in src/types.hpp. Shape: (num_worlds, max_agents, 14).

    Attributes:
        pos_x: Global x-coordinate of the agent.
        pos_y: Global y-coordinate of the agent.
        rotation_as_quaternion (4D float): Represents a quaternion, a 3D rotation.
        rotation_from_axis (1D float): Represents the angular distance
        from the x-axis (2D rotation).
        goal_x: Global x-coordinate of the goal.
        goal_y: Global y-coordinate of the goal.
        vehicle_length: Length of the agent's bounding box.
        vehicle_width: Width of the agent's bounding box.
        vehicle_height: Height of the agent's bounding box.
        id: Unique identifier of the agent.
    """

    def __init__(self, abs_self_obs_tensor: torch.Tensor):
        """Initializes the ego state with an observation tensor."""
        self.pos_x = abs_self_obs_tensor[:, :, 0]
        self.pos_y = abs_self_obs_tensor[:, :, 1]
        self.pos_z = abs_self_obs_tensor[:, :, 2]
        self.rotation_as_quaternion = abs_self_obs_tensor[:, :, 3:7]
        self.rotation_angle = abs_self_obs_tensor[:, :, 7]
        self.goal_x = abs_self_obs_tensor[:, :, 8]
        self.goal_y = abs_self_obs_tensor[:, :, 9]
        self.vehicle_length = abs_self_obs_tensor[:, :, 10] * AGENT_SCALE
        self.vehicle_width = abs_self_obs_tensor[:, :, 11] * AGENT_SCALE
        self.vehicle_height = abs_self_obs_tensor[:, :, 12]
        self.id = abs_self_obs_tensor[:, :, 13]

    @classmethod
    def from_tensor(
        cls,
        abs_self_obs_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        """Creates an GlobalEgoState from a tensor."""
        if backend == "torch":
            return cls(abs_self_obs_tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of the ego state tensor."""
        return self.pos_x.shape

    def restore_mean(self, mean_x, mean_y):
        """Reapplies the mean to revert back to the original coordinates.
        - self.pos_x and self.pos_y are modified in place are of shape (num_worlds, num_agents).
        - mean_x and mean_y are expected to be of shape (num_worlds, 1).
        """
        # Reshape the mean to broadcast
        mean_x_reshaped = mean_x.view(-1, 1)
        mean_y_reshaped = mean_y.view(-1, 1)

        self.pos_x += mean_x_reshaped
        self.pos_y += mean_y_reshaped


@dataclass
class PartnerObs:
    speed: torch.Tensor
    rel_pos_x: torch.Tensor
    rel_pos_y: torch.Tensor
    orientation: torch.Tensor
    vehicle_length: torch.Tensor
    vehicle_width: torch.Tensor
    vehicle_height: torch.Tensor
    agent_type: torch.Tensor
    ids: torch.Tensor

    """
    A dataclass that represents information about other agents in the
    scenario, as viewed from the perspective of the ego agent
    (in relative coordinates). Initialized from partner_obs_tensor (src/bindings). For details, see
    `PartnerObservations` in src/types.hpp. Shape: (num_worlds, num_agents, num_agents-1, 8).
    """

    def __init__(self, partner_obs_tensor: torch.Tensor, mask=None):
        """Initializes the partner observation from a tensor."""
        self.mask = mask
        if self.mask is not None:  # Used for training
            self.data = partner_obs_tensor[self.mask][:, :, :6]
            self.data[:, :, 4] *= AGENT_SCALE
            self.data[:, :, 5] *= AGENT_SCALE
        else:
            self.speed = partner_obs_tensor[:, :, :, 0].unsqueeze(-1)
            self.rel_pos_x = partner_obs_tensor[:, :, :, 1].unsqueeze(-1)
            self.rel_pos_y = partner_obs_tensor[:, :, :, 2].unsqueeze(-1)
            self.orientation = partner_obs_tensor[:, :, :, 3].unsqueeze(-1)
            self.vehicle_length = (
                partner_obs_tensor[:, :, :, 4].unsqueeze(-1) * AGENT_SCALE
            )
            self.vehicle_width = (
                partner_obs_tensor[:, :, :, 5].unsqueeze(-1) * AGENT_SCALE
            )
            self.vehicle_height = partner_obs_tensor[:, :, :, 6].unsqueeze(-1)
            self.agent_type = (
                partner_obs_tensor[:, :, :, 7].unsqueeze(-1).long()
            )
            self.ids = partner_obs_tensor[:, :, :, 8].unsqueeze(-1)

    @classmethod
    def from_tensor(
        cls,
        partner_obs_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
        mask=None,
    ):
        """Creates an PartnerObs from a tensor."""
        if backend == "torch":
            tensor = partner_obs_tensor.to_torch().clone().to(device)
            obj = cls(tensor, mask=mask)
            obj.norm = torch.tensor(
                [
                    constants.MAX_ORIENTATION_RAD,
                    constants.MAX_VEH_LEN,
                    constants.MAX_VEH_WIDTH,
                ],
                device=device,
            )
            return obj

        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the partner observation."""
        if self.mask is not None:
            self.data[:, :, 0] /= constants.MAX_SPEED
            normalize_min_max_inplace(
                tensor=self.data[:, :, 1],
                min_val=constants.MIN_REL_GOAL_COORD,
                max_val=constants.MAX_REL_GOAL_COORD,
            )
            normalize_min_max_inplace(
                tensor=self.data[:, :, 2],
                min_val=constants.MIN_REL_GOAL_COORD,
                max_val=constants.MAX_REL_GOAL_COORD,
            )
            self.data[:, :, 3:6] /= self.norm
        else:
            self.speed = self.speed / constants.MAX_SPEED
            self.rel_pos_x = normalize_min_max(
                tensor=self.rel_pos_x,
                min_val=constants.MIN_REL_GOAL_COORD,
                max_val=constants.MAX_REL_GOAL_COORD,
            )
            self.rel_pos_y = normalize_min_max(
                tensor=self.rel_pos_y,
                min_val=constants.MIN_REL_GOAL_COORD,
                max_val=constants.MAX_REL_GOAL_COORD,
            )
            self.orientation = self.orientation / constants.MAX_ORIENTATION_RAD
            self.vehicle_length = self.vehicle_length / constants.MAX_VEH_LEN
            self.vehicle_width = self.vehicle_width / constants.MAX_VEH_WIDTH
            self.vehicle_height = (
                self.vehicle_height / constants.MAX_VEH_HEIGHT
            )

    def one_hot_encode_agent_types(self):
        """One-hot encodes the agent types. This operation increases the
        number of features by 3.
        """
        self.agent_type = self.agent_type.squeeze(-1)
        # Map to classes 0-3
        self.agent_type[
            self.agent_type == int(madrona_gpudrive.EntityType.Vehicle)
        ] = 1
        self.agent_type[
            self.agent_type == int(madrona_gpudrive.EntityType.Pedestrian)
        ] = 2
        self.agent_type[
            self.agent_type == int(madrona_gpudrive.EntityType.Cyclist)
        ] = 3

        self.agent_type = torch.nn.functional.one_hot(
            self.agent_type, num_classes=4
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape: (num_worlds, num_agents, num_agents-1)."""
        return self.speed.shape


@dataclass
class LidarObs:
    """Dataclass representing the scenario view through LiDAR sensors.
        - Shape: (num_worlds, num_agents, 3, num_lidar_points, 4).
        - Axis 2 represents the agent samples, road edge samples, and road line samples.
        - Axis 3 represents the lidar points per type, which can be configured in src/consts.hpp as `numLidarSamples`.
        - Axis 4 represents the depth, type and x, y, values of the lidar points.
    Initialized from lidar_tensor (src/bindings).
    For details, see `Lidar` and `LidarSample` in src/types.hpp.
    """

    def __init__(self, lidar_tensor: torch.Tensor):
        self.all_lidar_samples = lidar_tensor
        self.agent_samples = lidar_tensor[:, :, 0, :, :]
        self.road_edge_samples = lidar_tensor[:, :, 1, :, :]
        self.road_line_samples = lidar_tensor[:, :, 2, :, :]

    @classmethod
    def from_tensor(
        cls,
        lidar_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        if backend == "torch":
            return cls(lidar_tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape: (num_worlds, num_agents, 3, num_lidar_points, 4)."""
        return self.all_lidar_samples.shape


@dataclass
class BevObs:
    """Dataclass representing the scenario view through LiDAR sensors.
        - Shape: (num_worlds, num_agents, 200, 200, num_classes).
    Initialized from bev_observation_tensor (src/bindings).
    For details, see `BevObservation` and `BevObservations` in src/types.hpp.
    """

    def __init__(self, bev_observation_tensor: torch.Tensor):
        self.bev_segmentation_map = bev_observation_tensor

    @classmethod
    def from_tensor(
        cls,
        bev_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        if backend == "torch":
            return cls(bev_tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape: (num_worlds, num_agents, resolution, resolution, 1)."""
        return self.bev_segmentation_map.shape

    def one_hot_encode_bev_map(self):
        """One-hot encodes the agent types. This operation increases the
        number of features by 10.
        """
        self.bev_segmentation_map = torch.nn.functional.one_hot(
            self.bev_segmentation_map.long(),
            num_classes=constants.NUM_MADRONA_ENTITY_TYPES,  # From size of Madrona EntityType
        )


@dataclass
class TrafficLightObs:
    """
    A dataclass that represents traffic light information in the scenario.
    This data struct contains the time series of traffic light information.
    It contains the state (unknown: 0, stop: 1, caution: 2, go: 3), position, and lane_ids.
    Initialized from tl_states_tensor (from Manager.trafficLightTensor()).
    For details, see `TrafficLightState` in src/types.hpp.
    Shape: (num_worlds, max_traffic_lights, num_timesteps * features).
    Attributes:
        state: The state of each traffic light (0=unknown, 1=stop, 2=caution, 3=go)
        pos_x: X-coordinate of the traffic light
        pos_y: Y-coordinate of the traffic light
        pos_z: Z-coordinate of the traffic light
        time_index: Time index of the traffic light state
        lane_id: Lane ID associated with the traffic light
        valid_mask: Boolean mask indicating valid traffic lights
    """

    state: torch.Tensor
    pos_x: torch.Tensor
    pos_y: torch.Tensor
    pos_z: torch.Tensor
    time_index: torch.Tensor
    lane_id: torch.Tensor
    valid_mask: torch.Tensor

    def __init__(
        self,
        tl_states_tensor: torch.Tensor,

    ):
        """Initializes the traffic light observation from a tensor."""
        traj_length = constants.LOG_TRAJECTORY_LEN

        # Calculate indices based on C++ struct layout:
        # laneId (1) + state[traj_length] + x[traj_length] + y[traj_length] + z[traj_length] + timeIndex[traj_length] + numStates (1)

        lane_id_end_idx = 1
        state_end_idx = lane_id_end_idx + traj_length
        pos_x_end_idx = state_end_idx + traj_length
        pos_y_end_idx = pos_x_end_idx + traj_length
        pos_z_end_idx = pos_y_end_idx + traj_length
        time_index_end_idx = pos_z_end_idx + traj_length

        # Extract fields according to C++ struct layout
        # See `TrafficLightState` in src/types.hpp for details
        self.lane_id = tl_states_tensor[:, :, 0]  # Single lane ID value
        self.state = tl_states_tensor[
            :, :, lane_id_end_idx:state_end_idx
        ]  # state[traj_length]
        self.pos_x = tl_states_tensor[
            :, :, state_end_idx:pos_x_end_idx
        ].float()  # x[traj_length]
        self.pos_y = tl_states_tensor[
            :, :, pos_x_end_idx:pos_y_end_idx
        ].float()  # y[traj_length]
        self.pos_z = tl_states_tensor[
            :, :, pos_y_end_idx:pos_z_end_idx
        ].float()  # z[traj_length]
        self.time_index = tl_states_tensor[
            :, :, pos_z_end_idx:time_index_end_idx
        ]  # timeIndex[traj_length]
        self.num_states = tl_states_tensor[
            :, :, time_index_end_idx
        ]  # Single numStates value

        # Create a valid mask based on numStates
        # Traffic lights are valid if they have numStates > 0
        self.valid_mask = self.num_states > 0

    @classmethod
    def from_tensor(
        cls,
        tl_states_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        """Creates a TrafficLightObs from a tensor.
        Args:
            tl_states_tensor: The traffic light state tensor from the simulation
            backend: Which backend to use ("torch" or "jax")
            device: The device to place tensors on
        Returns:
            A TrafficLightObs instance
        """
        if backend == "torch":
            tensor = tl_states_tensor.to_torch().clone().to(device)
            obj = cls(tensor)
            return obj
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the traffic light observation coordinates."""

        # Normalize position coordinates
        self.pos_x = normalize_min_max(
            tensor=self.pos_x,
            min_val=constants.MIN_REL_COORD,
            max_val=constants.MAX_REL_COORD,
        )
        self.pos_y = normalize_min_max(
            tensor=self.pos_y,
            min_val=constants.MIN_REL_COORD,
            max_val=constants.MAX_REL_COORD,
        )
        self.pos_z = normalize_min_max(
            tensor=self.pos_z,
            min_val=constants.MIN_Z_COORD,
            max_val=constants.MAX_Z_COORD,
        )
    def one_hot_encode_states(self):
        """One-hot encodes the traffic light states.
        Converts the state values to one-hot encoded vectors with 4 classes:
        0: Unknown
        1: Stop
        2: Caution
        3: Go
        """
        # Make sure values are in range 0-3
        state_clamped = torch.clamp(self.state, 0, 3)
        # One-hot encode
        self.state_onehot = torch.nn.functional.one_hot(
            state_clamped, num_classes=4
        ) * self.valid_mask.unsqueeze(-1)

        return self.state_onehot

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape: (num_worlds, max_traffic_lights, num_timesteps)."""
        return self.state.shape