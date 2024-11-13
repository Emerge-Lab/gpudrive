import torch
from dataclasses import dataclass
from pygpudrive.env import constants
from pygpudrive.utils.geometry import normalize_min_max
import gpudrive


class EgoState:
    """A class to represent the ego state of the agent.

    Attributes:
        speed: Speed of the agent in relative coordinates.
        vehicle_length: Length of the agent's bounding box.
        vehicle_width: Width of the agent's bounding box.
        rel_goal_x: Relative x-coordinate to the goal.
        rel_goal_y: Relative y-coordinate to the goal.
        is_collided: Whether the agent is in collision with another object.
        id: Unique identifier of the agent.
    """

    def __init__(self, self_obs_tensor: torch.Tensor):
        """Initializes the ego state with an observation tensor."""
        self.speed = self_obs_tensor[:, :, 0]
        self.vehicle_length = self_obs_tensor[:, :, 1]
        self.vehicle_width = self_obs_tensor[:, :, 2]
        self.rel_goal_x = self_obs_tensor[:, :, 3]
        self.rel_goal_y = self_obs_tensor[:, :, 4]
        self.is_collided = self_obs_tensor[:, :, 5]
        self.id = self_obs_tensor[:, :, 6]

    @classmethod
    def from_tensor(
        cls, self_obs_tensor: gpudrive.madrona.Tensor, backend="torch"
    ):
        """Creates an EgoState from a tensor."""
        if backend == "torch":
            return cls(self_obs_tensor.to_torch())  # Pass the entire tensor
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the ego state to be between -1 and 1."""
        self.speed = self.speed / constants.MAX_SPEED
        self.vehicle_length = self.vehicle_length / constants.MAX_VEH_LEN
        self.vehicle_width = self.vehicle_width / constants.MAX_VEH_WIDTH
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
        self.is_collided = self.is_collided
        self.id = self.id

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of the ego state tensor."""
        return self.speed.shape


@dataclass
class PartnerObs:
    speed: torch.Tensor
    rel_pos_x: torch.Tensor
    rel_pos_y: torch.Tensor
    orientation: torch.Tensor
    vehicle_length: torch.Tensor
    vehicle_width: torch.Tensor
    agent_type: torch.Tensor
    ids: torch.Tensor

    """
    A dataclass that represents information about other agents in the
    scenario, as viewed from the perspective of the ego agent.
    """

    def __init__(self, partner_obs_tensor: torch.Tensor):
        """Initializes the partner observation from a tensor."""
        self.speed = partner_obs_tensor[:, :, :, 0].unsqueeze(-1)
        self.rel_pos_x = partner_obs_tensor[:, :, :, 1].unsqueeze(-1)
        self.rel_pos_y = partner_obs_tensor[:, :, :, 2].unsqueeze(-1)
        self.orientation = partner_obs_tensor[:, :, :, 3].unsqueeze(-1)
        self.vehicle_length = partner_obs_tensor[:, :, :, 4].unsqueeze(-1)
        self.vehicle_width = partner_obs_tensor[:, :, :, 5].unsqueeze(-1)
        self.agent_type = partner_obs_tensor[:, :, :, 6].unsqueeze(-1)
        self.ids = partner_obs_tensor[:, :, :, 7].unsqueeze(-1)

    @classmethod
    def from_tensor(
        cls, partner_obs_tensor: gpudrive.madrona.Tensor, backend="torch"
    ):
        """Creates an EgoState from a tensor."""
        if backend == "torch":
            return cls(partner_obs_tensor.to_torch())
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the partner observation."""
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
        self.agent_type = self.agent_type.long()
        self.ids = self.ids

    def one_hot_encode_agent_types(self):
        """One-hot encodes the agent types. This operation increases the
        number of features by 3.
        """
        # TODO: Fix type in GPUDrive directly
        self.agent_type = self.agent_type.squeeze(-1)
        self.agent_type[
            self.agent_type == int(gpudrive.EntityType.Vehicle)
        ] = 1
        self.agent_type[
            self.agent_type == int(gpudrive.EntityType.Pedestrian)
        ] = 2
        self.agent_type[
            self.agent_type == int(gpudrive.EntityType.Cyclist)
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
    # TODO: Implement this class
    pass
