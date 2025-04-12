import os
from typing import Optional

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from PIL import Image

import os
import torch
import matplotlib
from typing import Tuple, Optional, List, Dict, Any, Union
from matplotlib.patches import Circle, Polygon, RegularPolygon

from gpudrive.visualize.color import ROAD_GRAPH_COLORS, ROAD_GRAPH_TYPE_NAMES

def img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_rgb()."""
    # Adjusted margins to better accommodate 3D plots
    fig.subplots_adjust(
        left=0.0,    # Reduce left margin
        bottom=0.0,  # Reduce bottom margin
        right=1.0,   # Extend to right edge
        top=1.0,     # Extend to top edge
        wspace=0.0, 
        hspace=0.0
    )
    
    # Force render
    fig.canvas.draw()
    
    # Convert to numpy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img


def save_img_as_png(img: np.ndarray, filename: str = "/tmp/img.png"):
    """Saves np image to disk."""
    outdir = os.path.dirname(filename)
    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(img).save(filename)


def plot_roadgraph_points(ax, observation_roadgraph, env_idx, agent_idx):
    """Plots the road graph points by their type, using names instead of type numbers."""

    # Extract road graph types and positions
    roadgraph_types = observation_roadgraph.type[env_idx, agent_idx, :]
    roadgraph_x = observation_roadgraph.x[env_idx, agent_idx, :]
    roadgraph_y = observation_roadgraph.y[env_idx, agent_idx, :]

    # Plot points by type, mapping types to names
    for road_type, color in ROAD_GRAPH_COLORS.items():
        # Filter points by road type
        idx = roadgraph_types == road_type
        if idx.sum() > 0:
            ax.plot(
                roadgraph_x[idx],
                roadgraph_y[idx],
                ".",  # Plot as dots
                color=color,
                label=ROAD_GRAPH_TYPE_NAMES.get(
                    road_type, f"Type {road_type}"
                ),
            )


def plot_numpy_bounding_boxes(
    ax: matplotlib.axes.Axes,
    bboxes: np.ndarray,
    color: np.ndarray,
    alpha: Optional[float] = 1.0,
    line_width_scale: float = 1.5,
    as_center_pts: bool = False,
    label: Optional[str] = None,
) -> None:
    """Plots multiple bounding boxes.

    Args:
      ax: Fig handles.
      bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
        yaw).
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
      as_center_pts: If set to True, bboxes will be drawn as center points,
        instead of full bboxes.
      label: String, represents the meaning of the color for different boxes.
    """
    if bboxes.ndim != 2 or bboxes.shape[1] != 5:
        raise ValueError(
            (
                "Expect bboxes rank 2, last dimension of bbox 5"
                " got{}, {}, {} respectively"
            ).format(bboxes.ndim, bboxes.shape[1], color.shape)
        )

    if as_center_pts:
        ax.plot(
            bboxes[:, 0],
            bboxes[:, 1],
            "o",
            color=color,
            ms=2,
            alpha=alpha,
            linewidth=1.7 * line_width_scale,
            label=label,
        )
    else:
        c = np.cos(bboxes[:, 4])
        s = np.sin(bboxes[:, 4])
        pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
        length, width = bboxes[:, 2], bboxes[:, 3]
        u = np.array((c, s))
        ut = np.array((s, -c))

        # Compute box corner coordinates.
        tl = pt + length / 2 * u - width / 2 * ut
        tr = pt + length / 2 * u + width / 2 * ut
        br = pt - length / 2 * u + width / 2 * ut
        bl = pt - length / 2 * u - width / 2 * ut

        # Compute heading arrow using center left/right/front.
        cl = pt - width / 2 * ut
        cr = pt + width / 2 * ut
        cf = pt + length / 2 * u

        # Draw bboxes.
        ax.plot(
            [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
            [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
            color=color,
            zorder=4,
            linewidth=1.7 * line_width_scale,
            alpha=alpha,
            label=label,
        )

        # Draw heading arrow.
        ax.plot(
            [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
            [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
            color=color,
            zorder=6,
            alpha=alpha,
            linewidth=1.5 * line_width_scale,
            label=label,
        )


def plot_bounding_box(
    ax: matplotlib.axes.Axes,
    center: Optional[Union[Tuple[float, float], torch.Tensor]],
    vehicle_length: Union[float, torch.Tensor],
    vehicle_width: Union[float, torch.Tensor],
    orientation: Union[float, torch.Tensor],
    color: str,
    alpha: Optional[float] = 1.0,
    label: Optional[str] = None,
) -> None:
    """Plots bounding boxes, supporting both single and multiple agents.

    Args:
        ax: Matplotlib Axes handle.
        center: Tuple (x, y) specifying a single bounding box center or
                a tensor of shape (num_agents, 2) with x, y positions for multiple agents.
        vehicle_length: Length of the bounding box (float or tensor of shape (num_agents,)).
        vehicle_width: Width of the bounding box (float or tensor of shape (num_agents,)).
        orientation: Orientation of the bounding box (float or tensor of shape (num_agents,)).
        color: Color for the bounding boxes.
        alpha: Transparency of the bounding boxes (0.0 to 1.0).
        label: Optional label for the bounding boxes (only used for single-agent plots).
    """
    if isinstance(center, torch.Tensor):
        # Multiple bounding boxes
        if center.shape[-1] != 2:
            raise ValueError(
                "Center tensor must have shape (num_agents, 2) for multiple bounding boxes."
            )

        num_agents = center.shape[0]
        for i in range(num_agents):
            cx, cy = center[i]
            length = vehicle_length[i].item()
            width = vehicle_width[i].item()
            angle = orientation[i].item()

            # Compute bounding box corners
            corners_x = [
                cx - length / 2,
                cx + length / 2,
                cx + length / 2,
                cx - length / 2,
                cx - length / 2,
            ]
            corners_y = [
                cy - width / 2,
                cy - width / 2,
                cy + width / 2,
                cy + width / 2,
                cy - width / 2,
            ]

            # Apply rotation
            rotated_corners = [
                (
                    (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle) + cx,
                    (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle) + cy,
                )
                for x, y in zip(corners_x, corners_y)
            ]

            rotated_corners_x, rotated_corners_y = zip(*rotated_corners)
            ax.plot(
                np.concatenate(
                    [rotated_corners_x]
                ),  # Use np.concatenate to fix the addition
                np.concatenate(
                    [rotated_corners_y]
                ),  # Use np.concatenate to fix the addition
                color=color,
                alpha=alpha,
                linestyle="-",
                linewidth=2,
                label=label if i == 0 else None,
            )
    else:
        # Single bounding box
        cx, cy = center
        corners_x = [
            cx - vehicle_length / 2,
            cx + vehicle_length / 2,
            cx + vehicle_length / 2,
            cx - vehicle_length / 2,
            cx - vehicle_length / 2,
        ]
        corners_y = [
            cy - vehicle_width / 2,
            cy - vehicle_width / 2,
            cy + vehicle_width / 2,
            cy + vehicle_width / 2,
            cy - vehicle_width / 2,
        ]

        # Apply rotation for single bounding box
        rotated_corners = [
            (
                (x - cx) * np.cos(orientation)
                - (y - cy) * np.sin(orientation)
                + cx,
                (x - cx) * np.sin(orientation)
                + (y - cy) * np.cos(orientation)
                + cy,
            )
            for x, y in zip(corners_x, corners_y)
        ]

        rotated_corners_x, rotated_corners_y = zip(*rotated_corners)
        ax.plot(
            np.concatenate([rotated_corners_x]),
            np.concatenate([rotated_corners_y]),
            color=color,
            alpha=alpha,
            linestyle="-",
            label=label,
            linewidth=2,
        )


def get_corners_polygon(x, y, length, width, orientation):
    """Calculate the four corners of a speed bump (can be any) polygon."""
    # Compute the direction vectors based on orientation
    # print(length)
    c = np.cos(orientation)
    s = np.sin(orientation)
    u = np.array((c, s))  # Unit vector along the orientation
    ut = np.array((-s, c))  # Unit vector perpendicular to the orientation

    # Center point of the speed bump
    pt = np.array([x, y])

    # corners
    tl = pt + (length / 2) * u - (width / 2) * ut
    tr = pt + (length / 2) * u + (width / 2) * ut
    br = pt - (length / 2) * u + (width / 2) * ut
    bl = pt - (length / 2) * u - (width / 2) * ut

    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


def get_stripe_polygon(
    x: float,
    y: float,
    length: float,
    width: float,
    orientation: float,
    index: int,
    num_stripes: int,
) -> np.ndarray:

    """Calculate the corners of a stripe within the speed bump polygon."""

    # Compute the direction vectors
    c = np.cos(orientation)
    s = np.sin(orientation)
    u = np.array([c, s])  # Unit vector along the orientation (lengthwise)
    ut = np.array([-s, c])  # Perpendicular unit vector (widthwise)

    # Total stripe height along the width
    stripe_width = length / num_stripes
    half_length = length / 2
    half_width = width / 2

    # Offset for the current stripe
    offset_start = -half_length + index * stripe_width
    offset_end = offset_start + stripe_width

    # Center of the speed bump
    center = np.array([x, y])

    # Calculate stripe corners
    stripe_corners = [
        center + u * offset_start + ut * half_width,  # Top-left
        center + u * offset_start - ut * half_width,  # Bottom-left
        center + u * offset_end - ut * half_width,  # Bottom-right
        center + u * offset_end + ut * half_width,  # Top-right
    ]

    return np.array(stripe_corners)


def plot_speed_bumps(
    x_coords: Union[float, np.ndarray],
    y_coords: Union[float, np.ndarray],
    segment_lengths: Union[float, torch.Tensor],
    segment_widths: Union[float, torch.Tensor],
    segment_orientations: Union[float, torch.Tensor],
    ax: matplotlib.axes.Axes,
    facecolor: str = None,
    edgecolor: str = None,
    alpha: float = None,
) -> None:
    facecolor = "xkcd:goldenrod"
    edgecolor = "xkcd:black"
    alpha = 0.5
    for x, y, length, width, orientation in zip(
        x_coords,
        y_coords,
        segment_lengths,
        segment_widths,
        segment_orientations,
    ):
        # method1: from waymax using hatch as diagonals
        points = get_corners_polygon(x, y, length, width, orientation)

        p = Polygon(
            points,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0,
            alpha=alpha,
            hatch=r"//",
            zorder=2,
        )

        ax.add_patch(p)

    pass


def plot_stop_sign(
    point: np.ndarray,
    ax: matplotlib.axes.Axes,
    radius: float = None,
    facecolor: str = None,
    edgecolor: str = None,
    linewidth: float = None,
    alpha: float = None,
) -> None:
    # Default configurations for the stop sign
    facecolor = "#c04000" if facecolor is None else facecolor
    edgecolor = "white" if edgecolor is None else edgecolor
    linewidth = 1.5 if linewidth is None else linewidth
    radius = 1.0 if radius is None else radius
    alpha = 1.0 if alpha is None else alpha

    point = np.array(point).reshape(-1)

    p = RegularPolygon(
        point,
        numVertices=6,  # For hexagonal stop sign
        radius=radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(p)


def plot_crosswalk(
    points,
    ax: plt.Axes = None,
    facecolor: str = None,
    edgecolor: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = (
        crosswalk_config["facecolor"] if facecolor is None else facecolor
    )
    edgecolor = (
        crosswalk_config["edgecolor"] if edgecolor is None else edgecolor
    )
    alpha = crosswalk_config["alpha"] if alpha is None else alpha

    p = Polygon(
        points,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
        alpha=alpha,
        hatch=r"//",
        zorder=1,
    )

    ax.add_patch(p)


def plot_numpy_bounding_boxes_multiple_policy(
    ax: matplotlib.axes.Axes,
    bboxes_s: List[np.ndarray],
    colors: List[np.ndarray],
    alpha: Optional[float] = 1.0,
    line_width_scale: float = 1.5,
    as_center_pts: bool = False,
    label: Optional[str] = None,
) -> None:
    """Plots multiple bounding boxes.

    Args:
      ax: Fig handles.
      bboxes_s: Shape (num_policies,bboxes)
      bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
        yaw).
      colors: (num_policies,color)
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
      as_center_pts: If set to True, bboxes will be drawn as center points,
        instead of full bboxes.
      label: String, represents the meaning of the color for different boxes.
    """

    for bboxes,color in zip(bboxes_s,colors):
        if bboxes.ndim != 2 or bboxes.shape[1] != 5:
            raise ValueError(
                (
                    "Expect bboxes rank 2, last dimension of bbox 5"
                    " got{}, {}, {} respectively"
                ).format(bboxes.ndim, bboxes.shape[1], color.shape)
            )

        if as_center_pts:
            ax.plot(
                bboxes[:, 0],
                bboxes[:, 1],
                "o",
                color=color,
                ms=2,
                alpha=alpha,
                linewidth=1.7 * line_width_scale,
                label=label,
            )
        else:
            c = np.cos(bboxes[:, 4])
            s = np.sin(bboxes[:, 4])
            pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
            length, width = bboxes[:, 2], bboxes[:, 3]
            u = np.array((c, s))
            ut = np.array((s, -c))

            # Compute box corner coordinates.
            tl = pt + length / 2 * u - width / 2 * ut
            tr = pt + length / 2 * u + width / 2 * ut
            br = pt - length / 2 * u + width / 2 * ut
            bl = pt - length / 2 * u - width / 2 * ut

            # Compute heading arrow using center left/right/front.
            cl = pt - width / 2 * ut
            cr = pt + width / 2 * ut
            cf = pt + length / 2 * u

            # Draw bboxes.
            ax.plot(
                [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
                [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
                color=color,
                zorder=4,
                linewidth=1.7 * line_width_scale,
                alpha=alpha,
                label=label,
            )

            # Draw heading arrow.
            ax.plot(
                [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
                [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
                color=color,
                zorder=4,
                alpha=alpha,
                linewidth=1.5 * line_width_scale,
                label=label,
            )