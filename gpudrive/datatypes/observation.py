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
        is_goal_reached: Whether the agent has reached its goal position.
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
            self.is_goal_reached = self_obs_tensor[:, 7]
            self.id = self_obs_tensor[:, 8]
            self.steer_angle = self_obs_tensor[:, 9]
        else:
            self.speed = self_obs_tensor[:, :, 0]
            self.vehicle_length = self_obs_tensor[:, :, 1] * AGENT_SCALE
            self.vehicle_width = self_obs_tensor[:, :, 2] * AGENT_SCALE
            self.vehicle_height = self_obs_tensor[:, :, 3]
            self.rel_goal_x = self_obs_tensor[:, :, 4]
            self.rel_goal_y = self_obs_tensor[:, :, 5]
            self.is_collided = self_obs_tensor[:, :, 6]
            self.is_goal_reached = self_obs_tensor[:, :, 7]
            self.id = self_obs_tensor[:, :, 8]
            self.steer_angle = self_obs_tensor[:, :, 9]

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
        self.steer_angle /= (torch.pi / 3)

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
        self.pos_xy = abs_self_obs_tensor[:, :, :2]
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
        - Shape: (num_worlds, num_agents, 3, num_lidar_samples, 4).
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
    def num_lidar_samples(self) -> int:
        """Number of lidar samples per type."""
        return self.all_lidar_samples.shape[3]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape: (num_worlds, num_agents, 3, num_lidar_samples, 4)."""
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
    It contains the state (unknown, stop, caution, go), position, 
    and lane ID across the 90 recorded timesteps.
    
    Initialized from tl_states_tensor (from Manager.trafficLightTensor()). 
    For details, see `TrafficLightState` in src/types.hpp.
    Shape: (num_worlds, max_traffic_lights, num_timesteps, 6).
    
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
    current_time: int = 0  # Default to the first timestep

    def __init__(self, tl_states_tensor: torch.Tensor, mask=None, current_time=0):
        """Initializes the traffic light observation from a tensor."""
        self.mask = mask
        self.current_time = current_time
        
        if self.mask is not None:  # Used for training
            self.data = tl_states_tensor[self.mask]
        else:
            # Traffic light state (0=unknown, 1=stop, 2=caution, 3=go)
            self.state = tl_states_tensor[:, :, :, 0].long()
            # Position coordinates
            self.pos_x = tl_states_tensor[:, :, :, 1]
            self.pos_y = tl_states_tensor[:, :, :, 2]
            self.pos_z = tl_states_tensor[:, :, :, 3]
            # Time index and lane ID
            self.time_index = tl_states_tensor[:, :, :, 4].long()
            self.lane_id = tl_states_tensor[:, :, :, 5].long()
            
            # Create a mask for valid traffic lights (lane_id != -1)
            self.valid_mask = (self.lane_id != -1).float()

    @classmethod
    def from_tensor(
        cls,
        tl_states_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
        mask=None,
        current_time=0,
    ):
        """Creates a TrafficLightObs from a tensor.
        
        Args:
            tl_states_tensor: The traffic light state tensor from the simulation
            backend: Which backend to use ("torch" or "jax")
            device: The device to place tensors on
            mask: Optional mask to apply to the tensor
            current_time: The current timestep to use (default: 0)
            
        Returns:
            A TrafficLightObs instance
        """
        if backend == "torch":
            tensor = tl_states_tensor.to_torch().clone().to(device)
            obj = cls(tensor, mask=mask, current_time=current_time)
            return obj
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the traffic light observation coordinates."""
        if self.mask is not None:
            # Normalize position coordinates if using mask
            normalize_min_max_inplace(
                tensor=self.data[:, :, :, 1],  # x coordinate
                min_val=constants.MIN_REL_COORD,
                max_val=constants.MAX_REL_COORD,
            )
            normalize_min_max_inplace(
                tensor=self.data[:, :, :, 2],  # y coordinate
                min_val=constants.MIN_REL_COORD,
                max_val=constants.MAX_REL_COORD,
            )
            normalize_min_max_inplace(
                tensor=self.data[:, :, :, 3],  # z coordinate
                min_val=constants.MIN_Z_COORD,
                max_val=constants.MAX_Z_COORD,
            )
        else:
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

    def get_current_timestep(self):
        """Returns the data for the current timestep only.
        
        Returns:
            A dict with the traffic light data for the current timestep
        """
        return {
            'state': self.state[:, :, self.current_time],
            'pos_x': self.pos_x[:, :, self.current_time],
            'pos_y': self.pos_y[:, :, self.current_time],
            'pos_z': self.pos_z[:, :, self.current_time],
            'time_index': self.time_index[:, :, self.current_time],
            'lane_id': self.lane_id[:, :, self.current_time],
            'valid_mask': self.valid_mask[:, :, self.current_time]
        }
    
    def set_current_time(self, time_idx):
        """Sets the current timestep for convenient access.
        
        Args:
            time_idx: The timestep index to use
        """
        max_time = self.state.shape[2] - 1
        self.current_time = min(max(0, time_idx), max_time)
    
    def get_stop_light_mask(self, time_idx=None):
        """Returns a mask of traffic lights in the stop state.
        
        Args:
            time_idx: Optional specific timestep (defaults to current_time)
            
        Returns:
            A tensor with 1.0 for stop lights and 0.0 for others
        """
        time_idx = self.current_time if time_idx is None else time_idx
        return (self.state[:, :, time_idx] == 1).float() * self.valid_mask[:, :, time_idx]
    
    def get_caution_light_mask(self, time_idx=None):
        """Returns a mask of traffic lights in the caution state.
        
        Args:
            time_idx: Optional specific timestep (defaults to current_time)
            
        Returns:
            A tensor with 1.0 for caution lights and 0.0 for others
        """
        time_idx = self.current_time if time_idx is None else time_idx
        return (self.state[:, :, time_idx] == 2).float() * self.valid_mask[:, :, time_idx]
    
    def get_go_light_mask(self, time_idx=None):
        """Returns a mask of traffic lights in the go state.
        
        Args:
            time_idx: Optional specific timestep (defaults to current_time)
            
        Returns:
            A tensor with 1.0 for go lights and 0.0 for others
        """
        time_idx = self.current_time if time_idx is None else time_idx
        return (self.state[:, :, time_idx] == 3).float() * self.valid_mask[:, :, time_idx]
    
    def predict_state_changes(self, future_window=10):
        """Analyzes when traffic lights will change state within a future window.
        
        Args:
            future_window: Number of timesteps to look ahead
            
        Returns:
            Dictionary with time-to-change predictions for each traffic light
        """
        current_states = self.state[:, :, self.current_time]
        time_to_change = torch.ones_like(current_states) * -1s
        
        # Look ahead to find when lights change
        max_time = min(self.current_time + future_window, self.state.shape[2])
        for t in range(self.current_time + 1, max_time):
            # Where the state changes and we haven't recorded a change yet
            changed = (self.state[:, :, t] != current_states) & (time_to_change == -1)
            # Record the time delta to the change
            time_to_change[changed] = t - self.current_time
        
        return time_to_change * self.valid_mask[:, :, self.current_time]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape: (num_worlds, max_traffic_lights, num_timesteps)."""
        return self.state.shape