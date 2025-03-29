"""Metrics to calculate the signed distance between objects."""

import jax
from jax import numpy as jnp

from waymax.utils import geometry
import numpy as np
from typing import Tuple
import torch
from torch import nn
from torch.autograd import Function


class OverlapReward(nn.Module):
    def __init__(
        self,
        clip=5.0,
        weight=1.0,
    ):
        super().__init__()
        self.clip = clip
        self.weight = weight

    def forward(
        self, traj_pred: torch.Tensor, c: dict, aoi: list = None, **kwargs
    ):
        """
        traj_pred: [B, A, T, D]
        c: dict
        """
        T = traj_pred.shape[2]
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

        A = traj_5dof.shape[1]

        # negative means collision
        signed_distance = ComputeOverlap.apply(traj_5dof, mask)  # [B, A, T, A]

        # valid mask
        valid = mask.unsqueeze(-1).repeat(1, 1, 1, A)  # [B, A, T, A]
        valid = valid * valid.transpose(1, 3)  # [B, A, T, A]
        signed_distance = torch.where(
            valid, signed_distance, self.clip
        )  # [B, A, T, A]

        # Ignore all distance larger than clip
        reward = signed_distance * (signed_distance < self.clip) * self.weight

        return reward


class OverlapRewardSimple(nn.Module):
    def __init__(self, clip=5.0, weight=1.0):
        super().__init__()

        self.clip = clip
        self.weight = weight

    def forward(self, traj_pred: torch.Tensor, c: dict, aoi=None, **kwargs):
        """
        traj_pred: [B, A, T, D]
        weight: [B, A, T]
        """
        mask = ~c["agents_mask"]  # [B, A]

        if aoi is not None:
            traj_pred = traj_pred[:, aoi]
            mask = mask[:, aoi]

        B, A, T, _ = traj_pred.shape

        valid = mask.unsqueeze(-1).repeat(1, 1, A)  # [B, A, A]
        valid = valid * valid.transpose(1, 2)  # [B, A, A]
        valid = valid.unsqueeze(-2).repeat(1, 1, T, 1)  # [B, A, T, A]

        traj_pred_xy = traj_pred[..., :2]  # [B, A, T, 2]

        traj_all = traj_pred_xy.unsqueeze(3).repeat(
            1, 1, 1, A, 1
        )  # [B, A, T, A, 2]
        traj_all_transpose = traj_all.detach().transpose(
            1, 3
        )  # [B, A, T, A, 2]

        distance = torch.norm(
            traj_all - traj_all_transpose, dim=-1
        )  # [B, A, T, A]

        self_interaction = (
            torch.eye(A, dtype=torch.bool)
            .unsqueeze(0)
            .unsqueeze(-2)
            .repeat(B, 1, T, 1)
        )  # [B, A, T, A]
        self_interaction = self_interaction.type_as(mask)

        distance = torch.where(
            self_interaction, self.clip, distance
        )  # remove self interaction
        distance = torch.where(
            valid, distance, self.clip
        )  # remove invalid objects

        # Ignore all distance larger than clip
        reward = distance * (distance < self.clip) * self.weight

        return reward


# Wrapper to pytorch function
class ComputeOverlap(Function):
    @staticmethod
    def forward(ctx, traj, mask):
        # Convert input tensor to JAX array
        traj_jax = jnp.array(traj.detach().cpu().numpy())
        mask_jax = jnp.array(mask.detach().cpu().numpy())

        # Call the JAX function
        signed_distance_jax = jax.vmap(
            jax.vmap(
                compute_overlap,
                in_axes=(1, 1),
                out_axes=1,
            ),
            in_axes=(0, 0),
        )(traj_jax, mask_jax)

        # signed_distance_jax[b, x1, t, x2] -> J(x1, x2| b, t)
        signed_distance = torch.from_numpy(
            np.array(signed_distance_jax)
        ).type_as(traj)

        # Save the JAX input and output for backward pass
        ctx.save_for_backward(traj, mask)

        return signed_distance  # [B, A, T, A]

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the gradient of the loss with respect to the output of forward (signed_distance)
        # [B, A, T, A, 1, 1], grad_output(b, x1, t, x2, 0, 0) -> dL/dJ(x1, x2|b, t)
        grad_output = grad_output[..., None, None]  # [B, A, T, A, 1]

        # Retrieve the saved JAX input and output
        (traj, mask) = ctx.saved_tensors

        # Convert input and grad_output tensors to JAX arrays
        traj_jax = jnp.array(traj.detach().cpu().numpy())
        mask_jax = jnp.array(mask.detach().cpu().numpy())

        # Compute the gradient using JAX
        # grad_traj_jax[b, x1, t, x2, x3, d] -> dJ(x1, x2|b, t)/d(x3=x1,d|b,t)
        (grad_traj_jax,) = jax.vmap(
            jax.vmap(
                jax.jacfwd(compute_overlap, argnums=(0,)),
                in_axes=(1, 1),
                out_axes=1,
            ),
            in_axes=(0, 0),
        )(traj_jax, mask_jax)

        # Convert grad_input JAX array to PyTorch tensor
        grad_traj = torch.from_numpy(np.array(grad_traj_jax)).type_as(
            grad_output
        )

        # [B, A, T, A, A, D],
        # grad_traj(b, x1, t, x2, x3, d) -> [dL/dJ(x1, x2|b, t)]*[dJ(x1, x2|b, t)/d(x3=x2,d|b,t)]
        grad_prod = grad_traj * grad_output

        # # [B, A, T, D], sum over x1 and x2
        # grad_output[b, x1, t, d] -> dL/dx1(d|b, t)
        grad = grad_prod.sum(dim=(1, 3)).transpose(1, 2)

        return grad, None  # grad_mask is None


"""'
Adapt from https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/geometry_utils.py
"""


@jax.jit
def minkowski_sum_of_box_and_box_points(
    box1_points: jax.Array, box2_points: jax.Array
) -> jax.Array:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    The last dimensions of the input and return store the x and y coordinates of
    the points. Both box1_points and box2_points needs to be stored in
    counter-clockwise order. Otherwise the function will return incorrect results
    silently.

    Args:
        box1_points: Tensor of vertices for box 1, with shape:
        (num_boxes, num_points_per_box, 2).
        box2_points: Tensor of vertices for box 2, with shape:
        (num_boxes, num_points_per_box, 2).

    Returns:
        The Minkowski sum of the two boxes, of size (num_boxes,
        num_points_per_box * 2, 2). The points will be stored in counter-clockwise
        order.
    """
    # NUM_BOX_1 = box1_points.shape[0]
    # NUM_BOX_2 = box2_points.shape[0]
    NUM_VERTICES_IN_BOX = box1_points.shape[1]
    assert NUM_VERTICES_IN_BOX == 4, "Only support boxes"
    # Hard coded order to pick points from the two boxes. This is a simplification
    # of the generic convex polygons case. For boxes, the adjacent edges are
    # always 90 degrees apart from each other, so the index of vertices can be
    # hard coded.
    point_order_1 = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
    point_order_2 = jnp.array([0, 1, 1, 2, 2, 3, 3, 0])

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(
        box1_points
    )
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(
        box2_points
    )

    # The cross-product of the unit vectors indicates whether the downmost edge
    # in box2 is pointing to the left side (the inward side of the resulting
    # Minkowski sum) of the downmost edge in box1. If this is the case, pick
    # points from box1 in the order `point_order_2`, and pick points from box2 in
    # the order of `point_order_1`. Otherwise, we switch the order to pick points
    # from the two boxes, pick points from box1 in the order of `point_order_1`,
    # and pick points from box2 in the order of `point_order_2`.
    # Shape: (num_boxes, 1)
    condition = (
        jnp.cross(downmost_box1_edge_direction, downmost_box2_edge_direction)
        >= 0.0
    )
    # box1_point_order of size [num_boxes, num_points_per_box * 2 = 8, 1].
    box1_point_order = jnp.where(condition, point_order_2, point_order_1)
    box1_point_order = jnp.expand_dims(box1_point_order, axis=-1)

    # Shift box1_point_order by box1_start_idx, so that the first index in
    # box1_point_order is the downmost vertex in the box.
    box1_point_order = jnp.mod(
        box1_point_order + box1_start_idx, NUM_VERTICES_IN_BOX
    )
    # Gather points from box1 in order.
    # ordered_box1_points is of size [num_boxes, num_points_per_box * 2, 2].
    ordered_box1_points = jnp.take_along_axis(
        box1_points, box1_point_order, axis=-2
    )

    # Gather points from box2 as well.
    box2_point_order = jnp.where(condition, point_order_1, point_order_2)
    box2_point_order = jnp.expand_dims(box2_point_order, axis=-1)
    box2_point_order = jnp.mod(
        box2_point_order + box2_start_idx, NUM_VERTICES_IN_BOX
    )
    ordered_box2_points = jnp.take_along_axis(
        box2_points, box2_point_order, axis=-2
    )

    minkowski_sum = ordered_box1_points + ordered_box2_points
    return minkowski_sum


@jax.jit
def _get_downmost_edge_in_box(box: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Note: We assume box edges are given in a counter-clockwise order, so that
    the edge which starts with the downmost vertex (i.e. the downmost edge) is
    uniquely identified.

    Args:
    box: (num_boxes, num_points_per_box, 2). The last dimension contains the x-y
        coordinates of corners in boxes.

    Returns:
    A tuple of two tensors:
        downmost_vertex_idx: The index of the downmost vertex, which is also the
        index of the downmost edge. Shape: (num_boxes, 1, 1).
        downmost_edge_direction: The tangent unit vector of the downmost edge,
        pointing in the counter-clockwise direction of the box.
        Shape: (num_boxes, 1, 2).
    """
    # The downmost vertex is the lowest in the y dimension.
    # Shape: (num_boxes, 1).

    NUM_BOX, NUM_VERTICES_IN_BOX, _ = box.shape
    assert NUM_VERTICES_IN_BOX == 4, "Only support boxes"
    downmost_vertex_idx = jnp.argmin(box[..., 1], axis=-1)[..., None, None]

    # Find the counter-clockwise point edge from the downmost vertex.
    edge_start_vertex = jnp.take_along_axis(box, downmost_vertex_idx, axis=1)
    # edge_start_vertex = box[np.arange(NUM_BOX), downmost_vertex_idx, :]
    edge_end_idx = jnp.mod(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = jnp.take_along_axis(box, edge_end_idx, axis=1)
    # edge_end_vertex = box[np.arange(NUM_BOX), edge_end_idx, :]

    # Compute the direction of this downmost edge.
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = jnp.linalg.norm(
        downmost_edge, axis=-1, keepdims=True
    )
    downmost_edge_direction = downmost_edge / downmost_edge_length
    return downmost_vertex_idx, downmost_edge_direction


@jax.jit
def _get_edge_info(
    polygon_points: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes properties about the edges of a polygon.

    Args:
        polygon_points: Tensor containing the vertices of each polygon, with
        shape (num_polygons, num_points_per_polygon, 2). Each polygon is assumed
        to have an equal number of vertices.

    Returns:
        tangent_unit_vectors: A unit vector in (x,y) with the same direction as
        the tangent to the edge. Shape: (num_polygons, num_points_per_polygon, 2).
        normal_unit_vectors: A unit vector in (x,y) with the same direction as
        the normal to the edge.
        Shape: (num_polygons, num_points_per_polygon, 2).
        edge_lengths: Lengths of the edges.
        Shape (num_polygons, num_points_per_polygon).
    """
    # Shift the polygon points by 1 position to get the edges.
    # Shape: (num_polygons, 1, 2).
    first_point_in_polygon = polygon_points[:, 0:1, :]
    # Shape: (num_polygons, num_points_per_polygon, 2).
    shifted_polygon_points = jnp.concatenate(
        [polygon_points[:, 1:, :], first_point_in_polygon], axis=-2
    )
    # Shape: (num_polygons, num_points_per_polygon, 2).
    edge_vectors = shifted_polygon_points - polygon_points

    # Shape: (num_polygons, num_points_per_polygon).
    edge_lengths = jnp.linalg.norm(edge_vectors, axis=-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    tangent_unit_vectors = edge_vectors / jnp.expand_dims(
        edge_lengths, axis=-1
    )
    # Shape: (num_polygons, num_points_per_polygon, 2).
    normal_unit_vectors = jnp.stack(
        [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], axis=-1
    )
    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


@jax.jit
def signed_distance_from_point_to_convex_polygon(
    query_points: jax.Array, polygon_points: jax.Array
) -> jax.Array:
    """Finds the signed distances from query points to convex polygons.

    Each polygon is represented by a 2d tensor storing the coordinates of its
    vertices. The vertices must be ordered in counter-clockwise order. An
    arbitrary number of pairs (point, polygon) can be batched on the 1st
    dimension.

    Note: Each polygon is associated to a single query point.

    Args:
        query_points: (2). The last dimension is the x and y
        coordinates of points.
        polygon_points: (batch_size, num_points_per_polygon, 2). The last
        dimension is the x and y coordinates of vertices.

    Returns:
        A tensor containing the signed distances of the query points to the
        polygons. Shape: (batch_size,).
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(
        polygon_points
    )

    # Expand the shape of `query_points` to (num_polygons, 1, 2), so that
    # it matches the dimension of `polygons_points` for broadcasting.
    # query_points = query_points[None, None, :]
    query_points = jnp.expand_dims(query_points, axis=(0, 1))

    # Compute query points to polygon points distances.
    # Shape (num_polygons, num_points_per_polygon, 2).
    vertices_to_query_vectors = query_points - polygon_points

    # Shape (num_polygons, num_points_per_polygon).
    vertices_distances = jnp.linalg.norm(vertices_to_query_vectors, axis=-1)

    # Query point to edge distances are measured as the perpendicular distance
    # of the point from the edge. If the projection of this point on to the edge
    # falls outside the edge itself, this distance is not considered (as there)
    # will be a lower distance with the vertices of this specific edge.

    # Make distances negative if the query point is in the inward side of the
    # edge. Shape: (num_polygons, num_points_per_polygon).
    edge_signed_perp_distances = jnp.sum(
        -normal_unit_vectors * vertices_to_query_vectors, axis=-1
    )

    # If `edge_signed_perp_distances` are all less than 0 for a
    # polygon-query_point pair, then the query point is inside the convex polygon.
    is_inside = jnp.all(edge_signed_perp_distances <= 0, axis=-1)

    # Project the distances over the tangents of the edge, and verify where the
    # projections fall on the edge.
    # Shape: (num_polygons, num_edges_per_polygon).
    projection_along_tangent = jnp.sum(
        tangent_unit_vectors * vertices_to_query_vectors, axis=-1
    )
    projection_along_tangent_proportion = (
        projection_along_tangent / edge_lengths
    )

    # Shape: (num_polygons, num_edges_per_polygon).
    is_projection_on_edge = jnp.logical_and(
        projection_along_tangent_proportion >= 0.0,
        projection_along_tangent_proportion <= 1.0,
    )

    # If the point projection doesn't lay on the edge, set the distance to inf.
    edge_perp_distances = jnp.abs(edge_signed_perp_distances)
    edge_distances = jnp.where(
        is_projection_on_edge, edge_perp_distances, np.inf
    )

    # Aggregate vertex and edge distances.
    # Shape: (num_polyons, 2 * num_edges_per_polygon).
    edge_and_vertex_distance = jnp.concatenate(
        [edge_distances, vertices_distances], axis=-1
    )
    # Aggregate distances per polygon and change the sign if the point lays inside
    # the polygon. Shape: (num_polygons,).
    min_distance = jnp.min(edge_and_vertex_distance, axis=-1)
    signed_distances = jnp.where(is_inside, -min_distance, min_distance)

    return signed_distances


@jax.jit
def compute_overlap(
    pose_5dof: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Computes the sigend distance between objects, negative means collision.

    Args:
        pose_5dof: The pose of the objects at the current time step.
            Shape: (num_objects, 5)
            Must be in the format (x, y, length, width, yaw)
        mask: The mask of the objects. Shape: (num_batch, num_objects, num_steps)
    Returns:
        The signed distance between objects. Shape: (num_batch, num_objects, num_steps, num_objects)
    """
    A, _ = pose_5dof.shape

    # Shape: (A, 4, 2)
    corners = geometry.corners_from_bboxes(pose_5dof)

    corners = jnp.expand_dims(corners, axis=1)  # Shape: (A, 1, 4, 2)
    # corners = current_traj.bbox_corners
    corners_all = corners.repeat(A, axis=1)  # Shape: (0:A, 1:A, 2:4, 3:2)
    corners_all_transpose = corners_all.transpose(
        (1, 0, 2, 3)
    )  # Shape: (A, A, 4, 2)
    corners_all_transpose = jax.lax.stop_gradient(corners_all_transpose)

    corners_all = corners_all.reshape(-1, 4, 2)
    corners_all_transpose = corners_all_transpose.reshape(-1, 4, 2)

    minkowski_diff = minkowski_sum_of_box_and_box_points(
        corners_all, -corners_all_transpose
    )
    # (A*A,)
    signed_distance = signed_distance_from_point_to_convex_polygon(
        np.array([0, 0]), minkowski_diff
    )

    signed_distance = signed_distance.reshape(A, A)

    # Remove self-interaction
    self_interaction = jnp.eye(
        A, dtype=jnp.bool_
    )  # Shape: (num_objects, num_objects)
    signed_distance = jnp.where(self_interaction, 1e3, signed_distance)

    # Remove Invalid objects
    valid = jnp.outer(mask, mask)
    valid = valid * ~self_interaction  # Shape: (num_objects, num_objects)
    signed_distance = jnp.where(valid, signed_distance, 1e3)

    return signed_distance  # Shape: (A, A)
