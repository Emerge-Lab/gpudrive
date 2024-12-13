import os
import torch
import math
import matplotlib
from typing import Tuple, Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import gpudrive
from pygpudrive.visualize import utils
from pygpudrive.datatypes.roadgraph import (
    LocalRoadGraphPoints,
    GlobalRoadGraphPoints,
)
from pygpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
)
from pygpudrive.datatypes.control import ResponseType
from pygpudrive.visualize.color import (
    ROAD_GRAPH_COLORS,
    ROAD_GRAPH_TYPE_NAMES,
    REL_OBS_OBJ_COLORS,
    AGENT_COLOR_BY_STATE,
)

OUT_OF_BOUNDS = 1000


class MatplotlibVisualizer:
    def __init__(
        self, sim_object, vis_config: Dict[str, Any], goal_radius, backend: str
    ):
        self.sim_object = sim_object
        self.vis_config = vis_config
        self.backend = backend
        self.device = "cpu"
        self.controlled_agents = self.get_controlled_agents_mask()
        self.goal_radius = goal_radius

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (
            (self.sim_object.controlled_state_tensor().to_torch() == 1)
            .squeeze(axis=2)
            .to(self.device)
        )

    def plot_simulator_state(
        self,
        env_indices: List[int],
        time_steps: Optional[List[int]] = None,
        center_agent_indices: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (15, 15),
        zoom_radius: int = 100,
        return_single_figure: bool = False,
    ):
        """
        Plot simulator states for one or multiple environments.

        Args:
            env_indices: List of environment indices to plot.
            time_steps: Optional list of time steps corresponding to each environment.
            center_agent_indices: Optional list of center agent indices for zooming.
            figsize: Tuple for figure size of each subplot.
            zoom_radius: Radius for zooming in around the center agent.
            return_single_figure: If True, plots all environments in a single figure.
                                Otherwise, returns a list of figures.
        """
        if not isinstance(env_indices, list):
            env_indices = [env_indices]  # Ensure env_indices is a list

        if time_steps is None:
            time_steps = [None] * len(env_indices)  # Default to None for all
        if center_agent_indices is None:
            center_agent_indices = [None] * len(
                env_indices
            )  # Default to None for all

        # Extract data for all environments
        global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim_object.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        global_agent_states = GlobalEgoState.from_tensor(
            self.sim_object.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        response_type = ResponseType.from_tensor(
            tensor=self.sim_object.response_type_tensor(),
            backend=self.backend,
            device=self.device,
        )

        agent_infos = self.sim_object.info_tensor().to_torch().to(self.device)

        figs = []  # Store all figures if returning multiple

        if return_single_figure:
            # Calculate rows and columns for square layout
            num_envs = len(env_indices)
            num_rows = math.ceil(math.sqrt(num_envs))
            num_cols = math.ceil(num_envs / num_rows)

            total_figsize = (figsize[0] * num_cols, figsize[1] * num_rows)
            fig, axes = plt.subplots(
                nrows=num_rows,
                ncols=num_cols,
                figsize=total_figsize,
                squeeze=False,
            )
            axes = axes.flatten()
        else:
            axes = [None] * len(
                env_indices
            )  # Placeholder for individual plotting

        # Calculate scale factors based on figure size
        max_fig_size = max(figsize)
        marker_scale = max_fig_size / 15  # Adjust this factor as needed
        line_width_scale = max_fig_size / 15  # Adjust this factor as needed

        # Iterate over each environment index
        for idx, (env_idx, time_step, center_agent_idx) in enumerate(
            zip(env_indices, time_steps, center_agent_indices)
        ):
            if return_single_figure:
                ax = axes[idx]
                ax.clear()  # Clear any previous plots
                ax.set_aspect("equal", adjustable="box")
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_aspect("equal", adjustable="box")
                ax.clear()
                figs.append(fig)

            # Get control mask and omit out-of-bound agents (dead agents)
            controlled = self.controlled_agents[env_idx, :]
            controlled_live = controlled & (
                torch.abs(global_agent_states.pos_x[env_idx, :]) < 1_000
            )

            is_offroad = (agent_infos[env_idx, :, 0] == 1) & controlled_live
            is_collided = (
                agent_infos[env_idx, :, 1:3].sum(axis=1) == 1
            ) & controlled_live
            is_ok = ~is_offroad & ~is_collided & controlled_live

            # Draw the road graph
            self._plot_roadgraph(
                road_graph=global_roadgraph,
                env_idx=env_idx,
                ax=ax,
                line_width_scale=line_width_scale,
                marker_size_scale=marker_scale,
            )

            # Draw the agents
            self._plot_filtered_agent_bounding_boxes(
                ax=ax,
                env_idx=env_idx,
                agent_states=global_agent_states,
                is_ok_mask=is_ok,
                is_offroad_mask=is_offroad,
                is_collided_mask=is_collided,
                response_type=response_type,
                alpha=1.0,
                line_width_scale=line_width_scale,
                marker_size_scale=marker_scale,
            )

            # Plot rollout statistics
            num_controlled = controlled.sum().item()
            num_off_road = is_offroad.sum().item()
            num_collided = is_collided.sum().item()
            if time_step is not None:
                ax.text(
                    0.5,  # Horizontal center
                    0.95,  # Vertical location near the top
                    f"$t$ = {time_step}  | $N_c$ = {num_controlled}; "
                    f"off-road: {num_off_road/num_controlled:.2f}; "
                    f"collision: {num_collided/num_controlled:.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=20 * marker_scale,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
                )

            # Determine center point for zooming
            if center_agent_idx is not None:
                center_x = global_agent_states.pos_x[
                    env_idx, center_agent_idx
                ].item()
                center_y = global_agent_states.pos_y[
                    env_idx, center_agent_idx
                ].item()
            else:
                center_x = 0  # Default center x-coordinate
                center_y = 0  # Default center y-coordinate

            # Set zoom window around the center
            ax.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
            ax.set_ylim(center_y - zoom_radius, center_y + zoom_radius)

            ax.set_xticks([])
            ax.set_yticks([])

        if return_single_figure:
            for ax in axes[len(env_indices) :]:
                ax.axis("off")  # Hide unused subplots
            plt.tight_layout()
            return fig
        else:
            return figs

    def _get_endpoints(self, x, y, length, yaw):
        """Compute the start and end points of a road segment."""
        center = np.array([x, y])
        start = center - np.array([length * np.cos(yaw), length * np.sin(yaw)])
        end = center + np.array([length * np.cos(yaw), length * np.sin(yaw)])
        return start, end

    def _plot_roadgraph(
        self,
        env_idx: int,
        road_graph: GlobalRoadGraphPoints,
        ax: matplotlib.axes.Axes,
        marker_size_scale: int = 1.0,
        line_width_scale: int = 1.0,
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
                    # Get coordinates and metadata
                    x_coords = road_graph.x[env_idx, road_mask].tolist()
                    y_coords = road_graph.y[env_idx, road_mask].tolist()
                    segment_lengths = road_graph.segment_length[
                        env_idx, road_mask
                    ].tolist()
                    segment_orientations = road_graph.orientation[
                        env_idx, road_mask
                    ].tolist()

                    # Compute and draw road edges using start and end points
                    for x, y, length, orientation in zip(
                        x_coords,
                        y_coords,
                        segment_lengths,
                        segment_orientations,
                    ):
                        start, end = self._get_endpoints(
                            x, y, length, orientation
                        )

                        if road_point_type == int(
                            gpudrive.EntityType.RoadEdge
                        ):
                            line_width = 1.1 * line_width_scale

                        else:
                            line_width = 0.75 * line_width_scale

                        ax.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            color=ROAD_GRAPH_COLORS[road_point_type],
                            linewidth=line_width,
                        )

                else:
                    # Dots for other road point types
                    ax.scatter(
                        road_graph.x[env_idx, road_mask],
                        road_graph.y[env_idx, road_mask],
                        s=5 * marker_size_scale,
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
        response_type: Any,
        alpha: Optional[float] = 1.0,
        as_center_pts: bool = False,
        label: Optional[str] = None,
        plot_goal_points: bool = True,
        line_width_scale: int = 1.0,
        marker_size_scale: int = 1.0,
    ) -> None:
        """Plots bounding boxes for agents filtered by environment index and mask.

        Args:
            ax: Matplotlib axis for plotting.
            global_agent_state: The global state of agents from `GlobalEgoState`.
            env_idx: Environment index to select specific environment agents.
            response_type: Mask to filter agents.
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
            color=AGENT_COLOR_BY_STATE["off_road"],
            alpha=alpha,
            line_width_scale=line_width_scale,
            as_center_pts=as_center_pts,
            label=label,
        )

        if plot_goal_points:
            goal_x = agent_states.goal_x[env_idx, is_offroad_mask].numpy()
            goal_y = agent_states.goal_y[env_idx, is_offroad_mask].numpy()
            ax.scatter(
                goal_x,
                goal_y,
                s=5 * marker_size_scale,
                linewidth=1.5 * line_width_scale,
                c=AGENT_COLOR_BY_STATE["off_road"],
                marker="x",
            )

            for x, y in zip(goal_x, goal_y):
                circle = Circle(
                    (x, y),
                    radius=self.goal_radius,
                    color=AGENT_COLOR_BY_STATE["off_road"],
                    fill=False,
                    linestyle="--",
                )
                ax.add_patch(circle)

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
            color=AGENT_COLOR_BY_STATE["collided"],
            alpha=alpha,
            line_width_scale=line_width_scale,
            as_center_pts=as_center_pts,
            label=label,
        )

        if plot_goal_points:
            goal_x = agent_states.goal_x[env_idx, is_collided_mask].numpy()
            goal_y = agent_states.goal_y[env_idx, is_collided_mask].numpy()
            ax.scatter(
                goal_x,
                goal_y,
                s=5 * marker_size_scale,
                c=AGENT_COLOR_BY_STATE["collided"],
                linewidth=1.5 * line_width_scale,
                marker="x",
            )

            for x, y in zip(goal_x, goal_y):
                circle = Circle(
                    (x, y),
                    radius=self.goal_radius,
                    color=AGENT_COLOR_BY_STATE["collided"],
                    fill=False,
                    linestyle="--",
                )
                ax.add_patch(circle)

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
            color=AGENT_COLOR_BY_STATE["ok"],
            alpha=alpha,
            line_width_scale=line_width_scale,
            as_center_pts=as_center_pts,
            label=label,
        )

        if plot_goal_points:
            goal_x = agent_states.goal_x[env_idx, is_ok_mask].numpy()
            goal_y = agent_states.goal_y[env_idx, is_ok_mask].numpy()
            ax.scatter(
                goal_x,
                goal_y,
                s=5 * marker_size_scale,
                linewidth=1.5 * line_width_scale,
                c=AGENT_COLOR_BY_STATE["ok"],
                marker="x",
            )

            for x, y in zip(goal_x, goal_y):
                circle = Circle(
                    (x, y),
                    radius=self.goal_radius,
                    color=AGENT_COLOR_BY_STATE["ok"],
                    fill=False,
                    linestyle="--",
                )
                ax.add_patch(circle)

        # Plot human_replay agents (those that are static or expert-controlled)
        log_replay = (
            response_type.static[env_idx, :] | response_type.moving[env_idx, :]
        ) & ~self.controlled_agents[env_idx, :]

        pos_x = agent_states.pos_x[env_idx, log_replay]
        pos_y = agent_states.pos_y[env_idx, log_replay]
        rotation_angle = agent_states.rotation_angle[env_idx, log_replay]
        vehicle_length = agent_states.vehicle_length[env_idx, log_replay]
        vehicle_width = agent_states.vehicle_width[env_idx, log_replay]

        # Define realistic bounds for log_replay agent positions
        valid_mask = (torch.abs(pos_x) < OUT_OF_BOUNDS) & (
            torch.abs(pos_y) < OUT_OF_BOUNDS
        )

        # Filter valid static agent attributes
        bboxes_static = np.stack(
            (
                pos_x[valid_mask].numpy(),
                pos_y[valid_mask].numpy(),
                vehicle_length[valid_mask].numpy(),
                vehicle_width[valid_mask].numpy(),
                rotation_angle[valid_mask].numpy(),
            ),
            axis=1,
        )

        # Plot static bounding boxes
        utils.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=bboxes_static,
            color=AGENT_COLOR_BY_STATE["log_replay"],
            alpha=alpha,
            line_width_scale=line_width_scale,
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
        figsize: Tuple[int, int] = (10, 10),
    ):
        """Plot observation from agent POV to inspect the information available to the agent.
        Args:
            agent_idx (int): Index of the agent whose observation is to be plotted.
            env_idx (int): Index of the environment in the batch.
            x_lim (Tuple[float, float], optional): x-axis limits. Defaults to (-100, 100).
            y_lim (Tuple[float, float], optional): y-axis limits. Defaults to (-100, 100).
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

        fig, ax = plt.subplots(figsize=figsize)
        ax.clear()  # Clear any previous plots
        ax.set_aspect("equal", adjustable="box")
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
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax
