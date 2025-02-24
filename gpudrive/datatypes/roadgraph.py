from dataclasses import dataclass
import torch
import enum
import madrona_gpudrive
from gpudrive.utils.geometry import normalize_min_max
from gpudrive.env import constants


class MapElementIds(enum.IntEnum):
    """Ids for different map elements to be mapped into a tensor to be consistent with
    https://github.com/waymo-research/waymax/blob/main/waymax/datatypes/roadgraph.py.

    These integers represent the ID of these specific types as defined in:
    https://waymo.com/open/data/motion/tfexample.
    """

    LANE_UNDEFINED = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    # Original definition skips 4.
    ROAD_LINE_UNKNOWN = 5
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_UNKNOWN = 14
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19
    DRIVEWAY = 20  # New datatype in v1.2.0: Driveway entrances
    UNKNOWN = -1


@dataclass
class GlobalRoadGraphPoints:
    """A class to represent global road graph points. All information is
    global but demeaned, that is, centered at zero.

    Attributes:
        x: x-coordinate of the road point.
        y: y-coordinate of the road point.
        segment_length: Length of the road segment.
        segment_width: Scale of the road segment.
        segment_heigth: Height of the road segment.
        orientation: Orientation of the road segment.
        type: Type of road point (e.g., intersection, straight road).
        id: Unique identifier of the road point (road id).
    """

    def __init__(self, roadgraph_tensor: torch.Tensor):
        """Initializes the global road graph points with a tensor."""
        self.x = roadgraph_tensor[:, :, 0]
        self.y = roadgraph_tensor[:, :, 1]
        self.segment_length = roadgraph_tensor[:, :, 2]
        self.segment_width = roadgraph_tensor[:, :, 3]
        self.segment_height = roadgraph_tensor[:, :, 4]
        self.orientation = roadgraph_tensor[:, :, 5]
        # Skipping the map element type for now (redundant with the map type).
        self.id = roadgraph_tensor[:, :, 7]
        # TODO: Use map type instead of enum (8 instead of 6)
        self.type = roadgraph_tensor[:, :, 6]

    @classmethod
    def from_tensor(
        cls,
        roadgraph_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        """Creates a GlobalRoadGraphPoints instance from a tensor."""
        if backend == "torch":
            return cls(roadgraph_tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def normalize(self):
        """Normalizes the road graph points to [-1, 1]."""
        self.x = normalize_min_max(
            self.x,
            min_val=constants.MIN_RG_COORD,
            max_val=constants.MAX_RG_COORD,
        )
        self.y = normalize_min_max(
            self.y,
            min_val=constants.MIN_RG_COORD,
            max_val=constants.MAX_RG_COORD,
        )
        self.segment_length = self.segment_length / constants.MAX_ROAD_SCALE
        self.segment_width = self.segment_width / constants.MAX_ROAD_SCALE
        self.segment_height = self.segment_height / constants.MAX_ROAD_SCALE
        self.orientation = self.orientation / constants.MAX_ORIENTATION_RAD
        self.id = self.id

    def one_hot_encode_road_point_types(self):
        """One-hot encodes the type of road point."""
        self.type = torch.nn.functional.one_hot(self.type, num_classes=21)

    def restore_mean(self, mean_x, mean_y):
        """Reapplies the mean to revert back to the original coordinates."""
        self.x += mean_x
        self.y += mean_y


@dataclass
class LocalRoadGraphPoints:
    """A class to represent local (relative) road graph points. Takes in
    `agent_roadmap_tensor`. Shape: (num_worlds, num_agents, num_road_points, 9).
    Note that num_road_points is set in src/consts.hpp and indicates the K
    closest road points to each agent (`kMaxAgentMapObservationsCount` in src/consts.hpp). The
    selection of these points is configured using `road_obs_algorithm`.

    Attributes:
        x: x-coordinate of the road point relative to each agent.
        y: y-coordinate of the road point relative to each agent.
        segment_length: Length of the road segment.
        segment_width: Scale of the road segment.
        segment_heigth: Height of the road segment.
        orientation: Orientation of the road segment.
        id: Unique identifier of the road point (road id).
        type: Type of road point (e.g., edge, lane).
    """

    def __init__(self, local_roadgraph_tensor: torch.Tensor, mask=None):
        """Initializes the global road graph points with a tensor."""
        self.mask = mask
        if self.mask is not None:
            local_roadgraph_tensor = local_roadgraph_tensor[mask]
            self.x = local_roadgraph_tensor[:, :, 0]
            self.y = local_roadgraph_tensor[:, :, 1]
            self.segment_length = local_roadgraph_tensor[:, :, 2]
            self.segment_width = local_roadgraph_tensor[:, :, 3]
            self.segment_height = local_roadgraph_tensor[:, :, 4]
            self.orientation = local_roadgraph_tensor[:, :, 5]
            self.id = local_roadgraph_tensor[:, :, 7]
            # Note: To use waymax map type take index 8 instead of 6
            self.data = local_roadgraph_tensor[:, :, :6]
            self.type = local_roadgraph_tensor[:, :, 6].long()
        else:
            self.x = local_roadgraph_tensor[:, :, :, 0]
            self.y = local_roadgraph_tensor[:, :, :, 1]
            self.segment_length = local_roadgraph_tensor[:, :, :, 2]
            self.segment_width = local_roadgraph_tensor[:, :, :, 3]
            self.segment_height = local_roadgraph_tensor[:, :, :, 4]
            self.orientation = local_roadgraph_tensor[:, :, :, 5]
            # Note: To use waymax map type take index 8 instead of 6
            self.type = local_roadgraph_tensor[:, :, :, 6].long()
            self.id = local_roadgraph_tensor[:, :, :, 7]
      
    @classmethod
    def from_tensor(
        cls,
        local_roadgraph_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
        mask=None,
    ):
        """Creates a GlobalRoadGraphPoints instance from a tensor."""
        if backend == "torch":
            tensor = local_roadgraph_tensor.to_torch().clone().to(device)
            obj = cls(tensor, mask=mask)
            obj.norm = torch.Tensor([
                constants.MAX_ROAD_LINE_SEGMENT_LEN,
                constants.MAX_ROAD_SCALE,
                constants.MAX_ROAD_SCALE,
                constants.MAX_ORIENTATION_RAD
            ]).to(device)
            return obj
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")
        
    def normalize(self):
        """Normalizes the road graph points to [-1, 1]."""
        
        if self.mask is not None:
            self.data[:, :, 0] = normalize_min_max(
                self.data[:, :, 0],
                min_val=constants.MIN_RG_COORD,
                max_val=constants.MAX_RG_COORD,
            )
            self.data[:, :, 1] = normalize_min_max(
                self.data[:, :, 1],
                min_val=constants.MIN_RG_COORD,
                max_val=constants.MAX_RG_COORD,
            )
            self.data[:, :, 2:6] /= self.norm
        else:
            self.x = normalize_min_max(
                self.x,
                min_val=constants.MIN_RG_COORD,
                max_val=constants.MAX_RG_COORD,
            )
            self.y = normalize_min_max(
                self.y,
                min_val=constants.MIN_RG_COORD,
                max_val=constants.MAX_RG_COORD,
            )
            self.segment_length = (
                self.segment_length / constants.MAX_ROAD_LINE_SEGMENT_LEN
            )
            self.segment_width = self.segment_width / constants.MAX_ROAD_SCALE
            self.segment_height = self.segment_height / constants.MAX_ROAD_SCALE
            self.orientation = self.orientation / constants.MAX_ORIENTATION_RAD

    def one_hot_encode_road_point_types(self):
        """One-hot encodes the type of road point."""
        self.type = torch.nn.functional.one_hot(self.type, num_classes=7)

    def shape(self):
        """Returns the shape of the local road graph tensor."""
        return self.x.shape
