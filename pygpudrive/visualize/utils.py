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

from pygpudrive.visualize.color import ROAD_GRAPH_COLORS, ROAD_GRAPH_TYPE_NAMES

def bg_img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_rgb()."""
    fig.subplots_adjust(
        left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0
    )
    fig.canvas.draw()
    
    # Extract image data
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # Close the figure
    return img

def img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_rgb()."""
    # Display xticks and yticks and title.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.9, wspace=0.0, hspace=0.0
    )
    fig.canvas.draw()
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
            zorder=4,
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
