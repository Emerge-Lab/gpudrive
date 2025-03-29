from dataclasses import dataclass
import numpy as np
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
    global but demeaned, that is, centered at zero. Takes in
    map_observation_tensor of shape (num_worlds, num_road_points, 9).

    Attributes:
        x: x-coordinate of the road point.
        y: y-coordinate of the road point.
        segment_length: Length of the road segment.
        segment_width: Scale of the road segment.
        segment_height: Height of the road segment.
        orientation: Orientation of the road segment.
        type: Type of road point (e.g., intersection, straight road).
        id: Unique identifier of the road point (road id).
    """

    def __init__(self, roadgraph_tensor: torch.Tensor):
        """Initializes the global road graph points with a tensor."""
        self.x = roadgraph_tensor[:, :, 0]
        self.y = roadgraph_tensor[:, :, 1]
        self.xy = torch.stack((self.x, self.y), dim=-1)
        self.segment_length = roadgraph_tensor[:, :, 2]
        self.segment_width = roadgraph_tensor[:, :, 3]
        self.segment_height = roadgraph_tensor[:, :, 4]
        self.orientation = roadgraph_tensor[:, :, 5]
        self.type = roadgraph_tensor[:, :, 6] # Original GPUDrive road types, used for plotting
        self.id = roadgraph_tensor[:, :, 7]
        self.vbd_type = roadgraph_tensor[:, :, 8] # VBD map types aligned with Waymax
        self.num_points = roadgraph_tensor.shape[1]

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
        # Reshape for broadcasting
        mean_x_reshaped = mean_x.view(-1, 1)
        mean_y_reshaped = mean_y.view(-1, 1)
        
        self.x += mean_x_reshaped
        self.y += mean_y_reshaped

    def restore_xy(self):
        """Shifts x, y from the midpoint to the starting point of a segment, along the heading angle."""
        self.x -= self.segment_length * np.cos(self.orientation)
        self.y -= self.segment_length * np.sin(self.orientation)

        # Get the dimensions
        num_worlds, num_road_points = self.x.shape
        device = self.x.device

        # Lists to collect new tensors for each batch
        new_x_batches = []
        new_y_batches = []
        new_segment_length_batches = []
        new_orientation_batches = []
        new_id_batches = []
        new_type_batches = []

        # Process each world in the batch
        for batch_idx in range(num_worlds):
            x_batch = self.x[batch_idx]
            y_batch = self.y[batch_idx]
            segment_length_batch = self.segment_length[batch_idx]
            orientation_batch = self.orientation[batch_idx]
            id_batch = self.id[batch_idx]
            type_batch = self.type[batch_idx]

            # Find the indices where ids change
            id_shifted = torch.cat(
                [id_batch[1:], id_batch.new_tensor([id_batch[-1] + 1])]
            )
            id_change = id_shifted != id_batch
            last_indices = torch.nonzero(id_change).squeeze(-1)

            # Lists to collect new tensors for the current batch
            new_x_list = []
            new_y_list = []
            new_segment_length_list = []
            new_orientation_list = []
            new_id_list = []
            new_type_list = []

            prev_idx = 0
            for idx in last_indices:
                idx = idx.item()
                # Get the slices up to idx+1 (inclusive)
                x_slice = x_batch[prev_idx : idx + 1]
                y_slice = y_batch[prev_idx : idx + 1]
                segment_length_slice = segment_length_batch[prev_idx : idx + 1]
                orientation_slice = orientation_batch[prev_idx : idx + 1]
                id_slice = id_batch[prev_idx : idx + 1]
                type_slice = type_batch[prev_idx : idx + 1]

                # Compute end_x and end_y for the last point in this id
                start_x_last = x_slice[-1]
                start_y_last = y_slice[-1]
                segment_length_last = segment_length_slice[-1]
                orientation_last = orientation_slice[-1]

                end_x = start_x_last + 2 * segment_length_last * torch.cos(
                    orientation_last
                )
                end_y = start_y_last + 2 * segment_length_last * torch.sin(
                    orientation_last
                )

                # Orientation is set to zero for the final point
                end_orientation = torch.tensor(0.0, device=device)
                # Segment length is zero for the final point
                end_segment_length = torch.tensor(0.0, device=device)
                # Id remains the same
                end_id = id_slice[-1]
                # Type remains the same
                end_type = type_slice[-1]

                # Append the slices and the new point to the lists
                new_x_list.append(x_slice)
                new_y_list.append(y_slice)
                new_segment_length_list.append(segment_length_slice)
                new_orientation_list.append(orientation_slice)
                new_id_list.append(id_slice)
                new_type_list.append(type_slice)

                # Append the new point
                new_x_list.append(end_x.unsqueeze(0))
                new_y_list.append(end_y.unsqueeze(0))
                new_segment_length_list.append(end_segment_length.unsqueeze(0))
                new_orientation_list.append(end_orientation.unsqueeze(0))
                new_id_list.append(end_id.unsqueeze(0))
                new_type_list.append(end_type.unsqueeze(0))

                prev_idx = idx + 1

            # Concatenate the lists to form the new tensors for the current batch
            new_x_batch = torch.cat(new_x_list)
            new_y_batch = torch.cat(new_y_list)
            new_segment_length_batch = torch.cat(new_segment_length_list)
            new_orientation_batch = torch.cat(new_orientation_list)
            new_id_batch = torch.cat(new_id_list)
            new_type_batch = torch.cat(new_type_list)

            # Ensure that the tensors have size num_points by padding or truncating
            total_points = new_x_batch.size(0)
            if total_points < self.num_points:
                # Pad with zeros to reach num_points
                pad_size = self.num_points - total_points
                pad_tensor = lambda t: torch.cat(
                    [t, torch.zeros(pad_size, device=device)]
                )
                new_x_batch = pad_tensor(new_x_batch)
                new_y_batch = pad_tensor(new_y_batch)
                new_segment_length_batch = pad_tensor(new_segment_length_batch)
                new_orientation_batch = pad_tensor(new_orientation_batch)
                new_id_batch = pad_tensor(new_id_batch)
                new_type_batch = pad_tensor(new_type_batch)
            elif total_points > self.num_points:
                # Truncate to num_points
                new_x_batch = new_x_batch[: self.num_points]
                new_y_batch = new_y_batch[: self.num_points]
                new_segment_length_batch = new_segment_length_batch[
                    : self.num_points
                ]
                new_orientation_batch = new_orientation_batch[
                    : self.num_points
                ]
                new_id_batch = new_id_batch[: self.num_points]
                new_type_batch = new_type_batch[: self.num_points]

            # Collect the new batch tensors
            new_x_batches.append(new_x_batch)
            new_y_batches.append(new_y_batch)
            new_segment_length_batches.append(new_segment_length_batch)
            new_orientation_batches.append(new_orientation_batch)
            new_id_batches.append(new_id_batch)
            new_type_batches.append(new_type_batch)

        # Stack the new tensors across the batch dimension
        self.x = torch.stack(new_x_batches, dim=0)
        self.y = torch.stack(new_y_batches, dim=0)
        self.xy = torch.stack((self.x, self.y), dim=-1)
        self.segment_length = torch.stack(new_segment_length_batches, dim=0)
        self.orientation = torch.stack(new_orientation_batches, dim=0)
        self.id = torch.stack(new_id_batches, dim=0)
        self.type = torch.stack(new_type_batches, dim=0)


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
        segment_height: Height of the road segment.
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
