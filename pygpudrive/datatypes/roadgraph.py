from dataclasses import dataclass
import torch
import enum
from utils.geometry import

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
class AgentRoadGraphPoints: # RoadGraphPoints in local reference frame for each agent
    x: torch.Tensor
    y: torch.Tensor
    segment_length: torch.Tensor
    scale_width: torch.Tensor
    scale_height: torch.Tensor
    orientation: torch.Tensor
    road_type: torch.Tensor
    ids: torch.Tensor

    @classmethod
    def from_tensor(cls, tensor):
        return cls(
            x=tensor[:, :, :, 0],
            y=tensor[:, :, :, 1],
            segment_length=tensor[:, :, :, 2],
            scale_width=tensor[:, :, :, 3],
            scale_height=tensor[:, :, :, 4],
            orientation=tensor[:, :, :, 5],
            road_type=tensor[:, :, :, 6],
        )

    def as_tensor(self):
        return torch.stack(
            [self.x, self.y, self.segment_length, self.scale_width,
             self.scale_height, self.orientation, self.road_type],
            dim=-1,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the road graph tensor."""
        return self.x.shape

    @property
    def x(self) -> torch.Tensor:
        """x location for all points."""
        return torch.stack(self.x)

    @property
    def y(self) -> torch.Tensor:
        """y location for all points."""
        return torch.stack(self.y)


@dataclass
class GlobalRoadGraphPoints: # RoadGraphPoints in global reference frame for each agent
    x: torch.Tensor
    y: torch.Tensor
    segment_length: torch.Tensor
    scale_width: torch.Tensor
    scale_height: torch.Tensor
    orientation: torch.Tensor
    road_type: torch.Tensor
    ids: torch.Tensor

    @classmethod
    def from_tensor(cls, tensor):
        return cls(
            x=tensor[:, :, :, 0],
            y=tensor[:, :, :, 1],
            segment_length=tensor[:, :, :, 2],
            scale_width=tensor[:, :, :, 3],
            scale_height=tensor[:, :, :, 4],
            orientation=tensor[:, :, :, 5],
            road_type=tensor[:, :, :, 6],
        )

    def as_tensor(self):
        return torch.stack(
            [self.x, self.y, self.segment_length, self.scale_width,
             self.scale_height, self.orientation, self.road_type],
            dim=-1,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the road graph tensor."""
        return self.x.shape

    @property
    def x(self) -> torch.Tensor:
        """x location for all points."""
        return torch.stack(self.x)

    @property
    def y(self) -> torch.Tensor:
        """y location for all points."""
        return torch.stack(self.y)
