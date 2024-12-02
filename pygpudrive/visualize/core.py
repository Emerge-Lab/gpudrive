import os
import torch
from pathlib import Path
import mediapy
import matplotlib
from typing import Tuple, Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import gpudrive
from pygpudrive.visualize import utils
from pygpudrive.datatypes.roadgraph import LocalRoadGraphPoints, GlobalRoadGraphPoints
from pygpudrive.datatypes.observation import LocalEgoState, GlobalEgoState, PartnerObs
from pygpudrive.datatypes.control import ControlMasks
from pygpudrive.visualize.color import (
    ROAD_GRAPH_COLORS,
    ROAD_GRAPH_TYPE_NAMES,
    REL_OBS_OBJ_COLORS,
)

OUT_OF_BOUNDS = 100

connect_points_thresholds = {
    int(gpudrive.EntityType.RoadEdge): 20,
    int(gpudrive.EntityType.RoadLine): 8,
    int(gpudrive.EntityType.RoadLane): 8,
}


class MatplotlibVisualizer:
    def __init__(self, sim_object, vis_config: Dict[str, Any], goal_radius, backend: str):
        self.sim_object = sim_object
        self.vis_config = vis_config
        self.backend = backend
        self.device = "cpu"
        self.controlled_agents = self.get_controlled_agents_mask()
        self.goal_radius = goal_radius

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (
            self.sim_object.controlled_state_tensor().to_torch() == 1
        ).squeeze(axis=2)

    def plot_simulator_state(
        self,
        env_idx: int,
        time_step: int=None,
        center_agent_idx=None,
        figsize: Tuple[int, int] = (15, 15),
        zoom_radius: int = 100,
    ):
        """Plot the current state of the simulator from a birds' eye view."""

        # Get global road graph (note: only needs to be done once)
        global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim_object.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # Get global agent state
        global_agent_states = GlobalEgoState.from_tensor(
            self.sim_object.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # Get control type tensor
        control_type = ControlMasks.from_tensor(
            tensor=self.sim_object.response_type_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # Get current state of agents
        agent_infos = self.sim_object.info_tensor().to_torch().to(self.device)

        # Get control mask and omit out of bound agents (dead agents)
        controlled = control_type.controlled[env_idx, :]
        controlled_live = controlled & (torch.abs(global_agent_states.pos_x[env_idx, :]) < 1_000)

        # fmt: off
        is_offroad = (agent_infos[env_idx, :, 0] == 1) & controlled_live
        is_collided = (agent_infos[env_idx, :, 1:3].sum(axis=1) == 1) & controlled_live
        is_ok = ~is_offroad & ~is_collided & controlled_live

        fig, ax = plt.subplots(figsize=figsize)

        # Draw the road graph
        self._plot_roadgraph(
            road_graph=global_roadgraph, env_idx=env_idx, ax=ax
        )

        # Draw the agents
        self._plot_filtered_agent_bounding_boxes(
            ax=ax,
            env_idx=env_idx,
            agent_states=global_agent_states,
            is_ok_mask=is_ok,
            is_offroad_mask=is_offroad,
            is_collided_mask=is_collided,
            control_type=control_type,
            alpha=1.0,
        )

        # Plot rollout statistics
        num_controlled = controlled.sum().item()
        num_off_road = is_offroad.sum().item()
        num_collided = is_collided.sum().item()
        if time_step is not None:
            ax.text(
                0.5,  # Horizontal center
                0.95,  # Vertical location near the top
                f"$t$ = {time_step}  | $N_c$ = {num_controlled}; off-road: {num_off_road/num_controlled:.2f}; collision: {num_collided/num_controlled:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=20,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),  # White background
            )

        # Determine center point for zooming
        if center_agent_idx is not None:
            center_x = global_agent_states.pos_x[env_idx, center_agent_idx].item()
            center_y = global_agent_states.pos_y[env_idx, center_agent_idx].item()
        else:
            center_x = 0  # Default center x-coordinate (e.g., origin)
            center_y = 0  # Default center y-coordinate (e.g., origin)

        # Set zoom window around the center
        ax.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
        ax.set_ylim(center_y - zoom_radius, center_y + zoom_radius)

        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax

    def _plot_roadgraph(
        self,
        env_idx: int,
        road_graph: GlobalRoadGraphPoints,
        ax: matplotlib.axes.Axes,
    ):
        """Plot the road graph."""

        for road_point_type in road_graph.type.unique().tolist():
            if road_point_type == int(gpudrive.EntityType._None):
                # Skip padding road points
                continue
            else:
                road_mask = road_graph.type[env_idx, :] == road_point_type

                if (
                    road_point_type == int(gpudrive.EntityType.RoadEdge)
                    or road_point_type == int(gpudrive.EntityType.RoadLine)
                    or road_point_type == int(gpudrive.EntityType.RoadLane)
                ):
                    # Get coordinates and IDs
                    x_coords = road_graph.x[env_idx, road_mask].tolist()
                    y_coords = road_graph.y[env_idx, road_mask].tolist()
                    ids = road_graph.id[env_idx, road_mask].tolist()

                    # Group points by ID
                    id_to_points = {}
                    for x, y, point_id in zip(x_coords, y_coords, ids):
                        if point_id not in id_to_points:
                            id_to_points[point_id] = []
                        id_to_points[point_id].append((x, y))

                    # Connect points within each road ID group
                    for point_id, points in id_to_points.items():
                        if len(points) > 1:
                            for i in range(len(points) - 1):
                                p1, p2 = points[i], points[i + 1]
                                distance = (
                                    (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
                                ) ** 0.5

                                if (
                                    distance
                                    <= connect_points_thresholds[
                                        road_point_type
                                    ]
                                ):
                                    ax.plot(
                                        [p1[0], p2[0]],
                                        [p1[1], p2[1]],
                                        color=ROAD_GRAPH_COLORS[
                                            road_point_type
                                        ],
                                        linewidth=1,
                                    )
                else:
                    ax.scatter(
                        road_graph.x[env_idx, road_mask],
                        road_graph.y[env_idx, road_mask],
                        s=2,
                        label=road_point_type,
                        color=ROAD_GRAPH_COLORS[int(road_point_type)],
                    )

    def _plot_filtered_agent_bounding_boxes(
        self,
        env_idx: int,
        ax: matplotlib.axes.Axes,
        agent_states: GlobalEgoState,
        is_ok_mask: torch.Tensor,
        is_offroad_mask: torch.Tensor,
        is_collided_mask: torch.Tensor,
        control_type: Any,
        alpha: Optional[float] = 1.0,
        as_center_pts: bool = False,
        label: Optional[str] = None,
        plot_goal_points: bool = True,
    ) -> None:
        """Plots bounding boxes for agents filtered by environment index and mask.

        Args:
            ax: Matplotlib axis for plotting.
            global_agent_state: The global state of agents from `GlobalEgoState`.
            env_idx: Environment index to select specific environment agents.
            control_type: Mask to filter agents.
            alpha: Alpha value for drawing, i.e., 0 means fully transparent.
            as_center_pts: If True, only plot center points instead of full boxes.
            label: Label for the plotted elements.
        """

        # Off-road agents
        bboxes_controlled_offroad = np.stack(
            (
                agent_states.pos_x[env_idx, is_offroad_mask].numpy(),
                agent_states.pos_y[env_idx, is_offroad_mask].numpy(),
                agent_states.vehicle_length[env_idx, is_offroad_mask].numpy(),
                agent_states.vehicle_width[env_idx, is_offroad_mask].numpy(),
                agent_states.rotation_angle[env_idx, is_offroad_mask].numpy(),
            ),
            axis=1,
        )

        utils.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=bboxes_controlled_offroad,
            color="orange",
            alpha=alpha,
            as_center_pts=as_center_pts,
            label=label,
        )

        # Collided agents
        bboxes_controlled_collided = np.stack(
            (
                agent_states.pos_x[env_idx, is_collided_mask].numpy(),
                agent_states.pos_y[env_idx, is_collided_mask].numpy(),
                agent_states.vehicle_length[env_idx, is_collided_mask].numpy(),
                agent_states.vehicle_width[env_idx, is_collided_mask].numpy(),
                agent_states.rotation_angle[env_idx, is_collided_mask].numpy(),
            ),
            axis=1,
        )

        utils.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=bboxes_controlled_collided,
            color="r",
            alpha=alpha,
            as_center_pts=as_center_pts,
            label=label,
        )

        # Living agents
        bboxes_controlled_ok = np.stack(
            (
                agent_states.pos_x[env_idx, is_ok_mask].numpy(),
                agent_states.pos_y[env_idx, is_ok_mask].numpy(),
                agent_states.vehicle_length[env_idx, is_ok_mask].numpy(),
                agent_states.vehicle_width[env_idx, is_ok_mask].numpy(),
                agent_states.rotation_angle[env_idx, is_ok_mask].numpy(),
            ),
            axis=1,
        )

        utils.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=bboxes_controlled_ok,
            color="g",
            alpha=alpha,
            as_center_pts=as_center_pts,
            label=label,
        )

        # Plot goal points for living agents
        if plot_goal_points:
            goal_x = agent_states.goal_x[env_idx, is_ok_mask].numpy()
            goal_y = agent_states.goal_y[env_idx, is_ok_mask].numpy()
            ax.scatter(
                goal_x,
                goal_y,
                s=5,
                c="g",
                marker="x",
            )

            for x, y in zip(goal_x, goal_y):
                circle = Circle((x, y), radius=self.goal_radius, color='g', fill=False, linestyle='--')
                ax.add_patch(circle)


        # Plot agents that are marked as static
        static = control_type.static[env_idx, :]

        pos_x = agent_states.pos_x[env_idx, static]
        pos_y = agent_states.pos_y[env_idx, static]
        rotation_angle = agent_states.rotation_angle[env_idx, static]
        vehicle_length = agent_states.vehicle_length[env_idx, static]
        vehicle_width = agent_states.vehicle_width[env_idx, static]

        # Define realistic bounds for static agent positions
        valid_static_mask = (torch.abs(pos_x) < OUT_OF_BOUNDS) & (
            torch.abs(pos_y) < OUT_OF_BOUNDS
        )

        # Filter valid static agent attributes
        bboxes_static = np.stack(
            (
                pos_x[valid_static_mask].numpy(),
                pos_y[valid_static_mask].numpy(),
                vehicle_length[valid_static_mask].numpy(),
                vehicle_width[valid_static_mask].numpy(),
                rotation_angle[valid_static_mask].numpy(),
            ),
            axis=1,
        )

        # Plot static bounding boxes
        utils.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=bboxes_static,
            color="darkslategray",
            alpha=alpha,
            as_center_pts=as_center_pts,
            label=label,
        )

    def plot_agent_observation(
        self,
        agent_idx: int,
        env_idx: int,
        observation_roadgraph: torch.Tensor = None,
        observation_ego: torch.Tensor = None,
        observation_partner: torch.Tensor = None,
        x_lim: Tuple[float, float] = (-100, 100),
        y_lim: Tuple[float, float] = (-100, 100),
        viz_config=None,
    ):
        """Plot observation from agent POV to inspect the information available to the agent.
        Args:
            agent_idx (int): Index of the agent whose observation is to be plotted.
            env_idx (int): Index of the environment in the batch.
            x_lim (Tuple[float, float], optional): x-axis limits. Defaults to (-100, 100).
            y_lim (Tuple[float, float], optional): y-axis limits. Defaults to (-100, 100).
            viz_config ([type], optional): Visualization config. Defaults to None.
        """
        observation_ego = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim_object.self_observation_tensor(),
            backend=self.backend,
            device="cpu",
        )

        observation_roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=self.sim_object.agent_roadmap_tensor(),
            backend=self.backend,
            device="cpu",
        )

        observation_partner = PartnerObs.from_tensor(
            partner_obs_tensor=self.sim_object.partner_observations_tensor(),
            backend=self.backend,
            device="cpu",
        )

        # Check if agent index is valid, otherwise return None
        if observation_ego.id[env_idx, agent_idx] == -1:
            return None, None

        viz_config = (
            utils.VizConfig()
            if viz_config is None
            else utils.VizConfig(**viz_config)
        )

        fig, ax = utils.init_fig_ax(viz_config)
        ax.set_title(f"obs agent: {agent_idx}")

        # Plot roadgraph if provided
        if observation_roadgraph is not None:
            for road_type, type_name in ROAD_GRAPH_TYPE_NAMES.items():
                mask = (
                    observation_roadgraph.type[env_idx, agent_idx, :]
                    == road_type
                )
                ax.scatter(
                    observation_roadgraph.x[env_idx, agent_idx, mask],
                    observation_roadgraph.y[env_idx, agent_idx, mask],
                    c=[ROAD_GRAPH_COLORS[road_type]],
                    s=7,
                    label=type_name,
                )

        # Plot partner agents if provided
        if observation_partner is not None:
            partner_positions = torch.stack(
                (
                    observation_partner.rel_pos_x[env_idx, agent_idx, :, :]
                    .squeeze()
                    .cpu(),
                    observation_partner.rel_pos_y[env_idx, agent_idx, :, :]
                    .squeeze()
                    .cpu(),
                ),
                dim=1,
            )  # Shape: (num_partners, 2)

            utils.plot_bounding_box(
                ax=ax,
                center=partner_positions,
                vehicle_length=observation_partner.vehicle_length[
                    env_idx, agent_idx, :, :
                ].squeeze(),
                vehicle_width=observation_partner.vehicle_width[
                    env_idx, agent_idx, :, :
                ].squeeze(),
                orientation=observation_partner.orientation[
                    env_idx, agent_idx, :, :
                ].squeeze(),
                color=REL_OBS_OBJ_COLORS["other_agents"],
                alpha=0.8,
            )

        if observation_ego is not None:
            ego_agent_color = (
                "darkred"
                if observation_ego.is_collided[env_idx, agent_idx]
                else REL_OBS_OBJ_COLORS["ego"]
            )
            utils.plot_bounding_box(
                ax=ax,
                center=(0, 0),
                vehicle_length=observation_ego.vehicle_length[
                    env_idx, agent_idx
                ].item(),
                vehicle_width=observation_ego.vehicle_width[
                    env_idx, agent_idx
                ].item(),
                orientation=0.0,
                color=ego_agent_color,
                alpha=1.0,
                label="Ego agent",
            )

            # Add an arrow for speed
            speed = observation_ego.speed[env_idx, agent_idx].item()
            ax.arrow(
                0,
                0,  # Start at the ego vehicle's position
                speed,
                0,  # Arrow points to the right, proportional to speed
                head_width=0.5,
                head_length=0.7,
                fc=REL_OBS_OBJ_COLORS["ego"],
                ec=REL_OBS_OBJ_COLORS["ego"],
            )

            ax.plot(
                observation_ego.rel_goal_x[env_idx, agent_idx],
                observation_ego.rel_goal_y[env_idx, agent_idx],
                markersize=23,
                label="Goal",
                marker="*",
                markeredgecolor="k",
                linestyle="None",
                color=REL_OBS_OBJ_COLORS["ego_goal"],
            )[0]

        # fig.legend(
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 0.1),
        #     ncol=5,
        #     fontsize=10,
        #     title="Elements",
        # )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        # ax.set_xticks([])
        # ax.set_yticks([])

        return fig, ax
