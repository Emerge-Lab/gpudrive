"""Metrics to calculate the signed distance of road map."""
import jax
from jax import numpy as jnp
from waymax import datatypes
import numpy as np
import torch
from torch import nn
from torch.autograd import Function


class OnroadReward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        traj_pred: torch.Tensor,
        c: dict,
        roadgraph_points: datatypes.RoadgraphPoints,
        weight: torch.Tensor = 0.1,
        aoi: list = None,
        **kwargs
    ):
        """
        traj_pred: [B, A, T, D]
        c: dict
        weight: [B, A, T]
        """
        T = traj_pred.shape[-2]
        traj_pred_xy = traj_pred[..., :2]
        traj_pred_yaw = traj_pred[..., 2:3]
        length = c["agents"][..., -1, 5:6].repeat(1, 1, T).unsqueeze(-1)
        width = c["agents"][..., -1, 6:7].repeat(1, 1, T).unsqueeze(-1)

        traj_5dof = torch.concatenate(
            [traj_pred_xy, length, width, traj_pred_yaw], dim=-1
        )

        mask = (~c["agents_mask"]).unsqueeze(-1).repeat(1, 1, T)  # [B, A, T]

        if aoi is not None:
            traj_5dof = traj_5dof[:, aoi]
            mask = mask[:, aoi]

        # negative means on road
        signed_distance = distance_offroad(
            traj_5dof, roadgraph_points
        )  # [B, A, T]

        # filter out already offroad
        signed_distance = signed_distance * (signed_distance[:, :, 0:1] < 0)

        # compute cost
        cost = torch.functional.F.relu(signed_distance)
        cost = cost * mask * weight

        return -cost


def distance_offroad(
    pose_5dof: torch.Tensor,
    roadgraph_points: datatypes.RoadgraphPoints,
) -> torch.Tensor:
    """Checks if the given trajectory is offroad.

    This determines the signed distance between each bounding box corner and the
    closest road edge (median or boundary). If the distance is negative, then the
    trajectory is onroad else offroad.

    Args:
        pose_5dof: Agent trajectories to test to see if they are on or off road of
        shape (B, A, T, D). The bounding boxes derived from center and shape
        of the trajectory will be used to determine if any point in the box is
        offroad.
        roadgraph_points: All of the roadgraph points in the run segment of shape
        (num_map_points). Roadgraph points of type `ROAD_EDGE_BOUNDARY` and
        `ROAD_EDGE_MEDIAN` are used to do the check.

    Returns:
        min_distances: a float array with the shape (..., num_objects). The object is offroad
        if the value is positive.
    """
    # ! We assume that all batches are on the same roadgraph
    # ! TODO: batched roadgraph

    # Shape: (num_batch, num_agents, num_steps, num_corners=4, 2).
    bbox_corners = corners_from_bboxes(pose_5dof)

    # ! Ignore z for now
    # Add in the Z dimension from the current center. This assumption will help
    # disambiguate between different levels of the roadgraph (i.e. under and over
    # passes).
    # Shape: (..., num_objects, 1, 1).
    # z = jnp.ones_like(bbox_corners[..., 0:1]) * trajectory.z[..., jnp.newaxis, :]
    # Shape: (..., num_objects, num_corners=4, 3).
    # bbox_corners = jnp.concatenate((bbox_corners, z), axis=-1)

    # [B, A]
    num_corners, dim = bbox_corners.shape[-2:]
    pre_dims = bbox_corners.shape[:-2]

    # Shape: (...*num_corners=4, 2).
    bbox_corners = torch.reshape(bbox_corners, [-1, dim])

    # Shape: (...*num_corners=4).
    distances, sign = compute_signed_distance_to_nearest_road_edge_point(
        bbox_corners, roadgraph_points
    )
    # Shape: (num_batches, num_agents, num_steps, num_map_points=4).
    distances = torch.reshape(distances, [*pre_dims, num_corners])
    signs = torch.reshape(sign, [*pre_dims, num_corners])

    signed_distance = distances * signs  # [B,A,T]
    max_distances, _ = torch.max(signed_distance, dim=-1)

    # Shape: (B, A, T).
    return max_distances


def compute_signed_distance_to_nearest_road_edge_point(
    query_points: torch.Tensor,
    roadgraph_points: datatypes.RoadgraphPoints,
    # z_stretch: float = 2.0,
) -> torch.Tensor:
    """Computes the signed distance from a set of queries to roadgraph points.

    Args:
    query_points: A set of query points for the metric of shape
        (num_query_points, 2).
    roadgraph_points: A set of roadgraph points of shape (num_map_points).
    z_stretch: Tolerance in the z dimension which determines how close to
        associate points in the roadgraph. This is used to fix problems with
        overpasses.

    Returns:
    Signed distances of the query points with the closest road edge points of
        shape (num_query_points). If the value is negative, it means that the
        actor is on the correct side of the road, if it is positive, it is
        considered `offroad`.
    """
    # Shape: (..., num_map_points, 2).
    # extract information from roadgraph_points
    # Do not consider invalid points.
    # Shape: (num_map_points).
    is_road_edge = datatypes.is_road_edge(roadgraph_points.types)
    is_valid = roadgraph_points.valid & is_road_edge

    sampled_points = np.asarray(roadgraph_points.xy)
    dir_xy = np.asarray(roadgraph_points.dir_xy)
    id = np.asarray(roadgraph_points.ids)

    # Filter out the invalid points
    sampled_points = sampled_points[is_valid, :]
    dir_xy = dir_xy[is_valid, :]
    id = id[is_valid]

    # Make them all in torch
    sampled_points_torch = torch.from_numpy(sampled_points).type_as(
        query_points
    )
    dir_xy_torch = torch.from_numpy(dir_xy).type_as(query_points)
    id_torch = torch.from_numpy(id).to(query_points.device)

    # Shape: (num_query_points, num_map_points, 2).
    differences = sampled_points_torch - query_points[:, None, :]

    # !Ignore the over/underpasses
    # Stretch difference in altitude to avoid over/underpasses.
    # z_stretched_differences = differences * jnp.array([[[1.0, 1.0, z_stretch]]])
    # square_distances = jnp.sum(z_stretched_differences**2, axis=-1)

    # Shape: (num_query_points, num_map_points).
    square_distances = torch.sum(differences**2, dim=-1)

    # Shape: (num_query_points).
    nearest_indices = torch.argmin(square_distances, dim=-1)
    prior_indices = torch.clip(nearest_indices - 1, min=0, max=None)

    nearest_xys = sampled_points_torch[nearest_indices, :]

    # Direction of the road edge at the nearest points. Should be normed and
    # tangent to the road edge.
    # Shape: (num_map_points, 2).
    nearest_vector_xys = dir_xy_torch[nearest_indices, :]

    # Direction of the road edge at the points that precede the nearest points.
    # Shape: (num_map_points, 2).
    prior_vector_xys = dir_xy_torch[prior_indices, :]

    # Shape: (num_query_points, 2).
    points_to_edge = query_points - nearest_xys

    # Get the signed distance to the half-plane boundary with a cross product.
    cross_product = cross_2d(points_to_edge, nearest_vector_xys)
    cross_product_prior = cross_2d(points_to_edge, prior_vector_xys)

    # If the prior point is contiguous, consider both half-plane distances.
    # Shape: (num_map_points).
    prior_point_in_same_curve = (
        id_torch[nearest_indices] == id_torch[prior_indices]
    )

    offroad_sign = torch.sign(
        torch.where(
            torch.logical_and(
                prior_point_in_same_curve, cross_product_prior < cross_product
            ),
            cross_product_prior,
            cross_product,
        )
    )

    # ! Hard code make all zeros to be positive

    offroad_sign = torch.where(
        offroad_sign == 0, torch.ones_like(offroad_sign), offroad_sign
    )
    # Shape: (num_query_points).
    return torch.norm(nearest_xys - query_points, dim=-1), offroad_sign


def corners_from_bboxes(bbox: torch.Tensor) -> torch.Tensor:
    """
    Computes corners for one 5 dof bbox.
    Args:
        bbox: [..., 5]
    Returns:
        points: [..., 4, 2]
    """
    # bbox: [..., 5]
    c, s = torch.cos(bbox[..., 4]), torch.sin(bbox[..., 4])
    lc = bbox[..., 2] / 2 * c
    ls = bbox[..., 2] / 2 * s
    wc = bbox[..., 3] / 2 * c
    ws = bbox[..., 3] / 2 * s

    dx = torch.stack([lc + ws, lc - ws, -lc - ws, -lc + ws], dim=-1)
    dy = torch.stack([ls - wc, ls + wc, -ls + wc, -ls - wc], dim=-1)
    # [..., 2]
    points = torch.stack([dx, dy], dim=-1)

    points += bbox[..., None, :2]

    return points


def cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
