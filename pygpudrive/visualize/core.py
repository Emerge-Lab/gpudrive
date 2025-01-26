import torch
import matplotlib
from typing import Tuple, Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
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
from pygpudrive.datatypes.trajectory import LogTrajectory
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
        self,
        sim_object,
        controlled_agent_mask,
        goal_radius,
        backend: str,
        num_worlds: int,
        render_config: Dict[str, Any],
        env_config: Dict[str, Any],
    ):
        self.sim_object = sim_object
        self.backend = backend
        self.device = "cpu"
        self.goal_radius = goal_radius
        self.num_worlds = num_worlds
        self.render_config = render_config
        self.figsize = (15, 15)
        self.env_config = env_config
        self.initialize_static_scenario_data(controlled_agent_mask)

    def initialize_static_scenario_data(self, controlled_agent_mask):
        """
        Initialize key information for visualization based on the
        current batch of scenarios.
        """
        self.response_type = ResponseType.from_tensor(
            tensor=self.sim_object.response_type_tensor(),
            backend=self.backend,
            device=self.device,
        )
        self.global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim_object.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        self.controlled_agent_mask = controlled_agent_mask.to(self.device)

        self.log_trajectory = LogTrajectory.from_tensor(
            self.sim_object.expert_trajectory_tensor(),
            self.num_worlds,
            self.controlled_agent_mask.shape[1],
            backend=self.backend,
        )

        # Cache pre-rendered road graphs for all environments
        # self.cached_roadgraphs = []
        # for env_idx in range(self.controlled_agent_mask.shape[0]):
        #     fig, ax = plt.subplots(figsize=self.figsize)
        #     self._plot_roadgraph(
        #         road_graph=self.global_roadgraph,
        #         env_idx=env_idx,
        #         ax=ax,
        #         line_width_scale=1.0,
        #         marker_size_scale=1.0,
        #     )
        #     self.cached_roadgraphs.append(fig)
        #     plt.close(fig)

    def plot_simulator_state(
        self,
        env_indices: List[int],
        time_steps: Optional[List[int]] = None,
        center_agent_indices: Optional[List[int]] = None,
        zoom_radius: int = 100,
        results_df: Optional[pd.DataFrame] = None,
        plot_log_replay_trajectory: bool = False,
        eval_mode: bool = False,
        agent_positions: Optional[torch.Tensor] = None,
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

        # Changes at every time step
        global_agent_states = GlobalEgoState.from_tensor(
            self.sim_object.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        agent_infos = (
            self.sim_object.info_tensor().to_torch().clone().to(self.device)
        )

        figs = []

        # Calculate scale factors based on figure size
        marker_scale = max(self.figsize) / 15
        line_width_scale = max(self.figsize) / 15

        # Iterate over each environment index
        for idx, (env_idx, time_step, center_agent_idx) in enumerate(
            zip(env_indices, time_steps, center_agent_indices)
        ):

            # Initialize figure and axes from cached road graph
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.clear()  # Clear any existing content
            ax.set_aspect("equal", adjustable="box")
            figs.append(fig)  # Add the new figure
            plt.close(fig)  # Close the figure to prevent carryover

            # Render the pre-cached road graph for the current environment
            # cached_roadgraph_array = utils.bg_img_from_fig(self.cached_roadgraphs[env_idx])
            # ax.imshow(
            #     cached_roadgraph_array,
            #     origin="upper",
            #     extent=(-100, 100, -100, 100),  # Stretch to full plot
            #     zorder=0,  # Draw as background
            # )

            # Explicitly set the axis limits to match your coordinates
            # cached_ax.set_xlim(-100, 100)
            # cached_ax.set_ylim(-100, 100)

            # Get control mask and omit out-of-bound agents (dead agents)
            controlled = self.controlled_agent_mask[env_idx, :]
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
                road_graph=self.global_roadgraph,
                env_idx=env_idx,
                ax=ax,
                line_width_scale=line_width_scale,
                marker_size_scale=marker_scale,
            )

            if plot_log_replay_trajectory:
                self._plot_log_replay_trajectory(
                    ax=ax,
                    control_mask=controlled_live,
                    env_idx=env_idx,
                    log_trajectory=self.log_trajectory,
                    line_width_scale=line_width_scale,
                )

            # Draw the agents
            self._plot_filtered_agent_bounding_boxes(
                ax=ax,
                env_idx=env_idx,
                agent_states=global_agent_states,
                is_ok_mask=is_ok,
                is_offroad_mask=is_offroad,
                is_collided_mask=is_collided,
                response_type=self.response_type,
                alpha=1.0,
                line_width_scale=line_width_scale,
                marker_size_scale=marker_scale,
            )

            if agent_positions is not None:
                # agent_positions shape is [num_worlds, max_agent_count, episode_len, 2]
                for agent_idx in range(agent_positions.shape[1]):
                    if controlled_live[agent_idx]:
                        # Plot trajectory for this agent
                        trajectory = agent_positions[env_idx, agent_idx, :time_step, :]  # Gets both x,y

                        # Filter out zeros and out of bounds values
                        valid_mask = ((trajectory[:, 0] != 0) & (trajectory[:, 1] != 0) & 
                                    (torch.abs(trajectory[:, 0]) < OUT_OF_BOUNDS) & 
                                    (torch.abs(trajectory[:, 1]) < OUT_OF_BOUNDS))
                        
                        ax.plot(
                            trajectory[valid_mask, 0].cpu(),  # x coordinates
                            trajectory[valid_mask, 1].cpu(),  # y coordinates
                            color='green',
                            alpha=0.5,
                            linewidth=2.5,
                            linestyle='-',  # solid line, use '--' for dashed or ':' for dotted
                        )


            if eval_mode and results_df is not None:

                    num_controlled = results_df.iloc[
                        env_idx
                    ].controlled_agents_in_scene
                    off_road_rate = results_df.iloc[env_idx].off_road * 100
                    collision_rate = results_df.iloc[env_idx].collided * 100
                    goal_rate = results_df.iloc[env_idx].goal_achieved * 100
                    other = results_df.iloc[env_idx].not_goal_nor_crashed * 100

                    ax.text(
                        0.5,  # Horizontal center
                        0.95,  # Vertical location near the top
                        f"t = {time_step} | $N_c$ = {num_controlled}; "
                        f"OR: {off_road_rate:.1f}; "
                        f"CR: {collision_rate:.1f}; "
                        f"GR: {goal_rate:.1f}; "
                        f"Other: {other:.1f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        fontsize=20 * marker_scale,
                        color="black",
                        bbox=dict(
                            facecolor="white", edgecolor="none", alpha=0.9
                        ),
                    )
                    
            else:
                # Plot rollout statistics
                num_controlled = controlled.sum().item()
                num_off_road = is_offroad.sum().item()
                num_collided = is_collided.sum().item()
                off_road_rate = (
                    num_off_road / num_controlled if num_controlled > 0 else 0
                )
                collision_rate = (
                    num_collided / num_controlled if num_controlled > 0 else 0
                )

                ax.text(
                    0.5,  # Horizontal center
                    0.95,  # Vertical location near the top
                    f"$t$ = {time_step}  | $N_c$ = {num_controlled}; "
                    f"off-road: {off_road_rate:.2f}; "
                    f"collision: {collision_rate:.2f}",
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

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

        return figs

    def _plot_log_replay_trajectory(
        self,
        ax: matplotlib.axes.Axes,
        env_idx: int,
        control_mask: torch.Tensor,
        log_trajectory: LogTrajectory,
        line_width_scale: int = 1.0,
    ):
        ax.scatter(
            log_trajectory.pos_xy[env_idx, control_mask, :, 0].numpy(),
            log_trajectory.pos_xy[env_idx, control_mask, :, 1].numpy(),
            color="lightgreen",
            linewidth=0.35 * line_width_scale,
            alpha=0.35,
            zorder=0,
        )

    def _get_endpoints(self, x, y, length, yaw):
        """Compute the start and end points of a road segment."""
        center = np.array([x, y])
        start = center - np.array([length * np.cos(yaw), length * np.sin(yaw)])
        end = center + np.array([length * np.cos(yaw), length * np.sin(yaw)])
        return start, end

    def _get_corners_polygon(self, x, y, length, width, orientation):
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
                    or road_point_type == int(gpudrive.EntityType.SpeedBump)
                    or road_point_type == int(gpudrive.EntityType.StopSign)
                    or road_point_type == int(gpudrive.EntityType.CrossWalk)
                ):
                    # Get coordinates and metadata
                    x_coords = road_graph.x[env_idx, road_mask].tolist()
                    y_coords = road_graph.y[env_idx, road_mask].tolist()
                    segment_lengths = road_graph.segment_length[
                        env_idx, road_mask
                    ].tolist()
                    segment_widths = road_graph.segment_width[
                        env_idx, road_mask
                    ].tolist()
                    segment_orientations = road_graph.orientation[
                        env_idx, road_mask
                    ].tolist()

                    if (
                        road_point_type == int(gpudrive.EntityType.RoadEdge)
                        or road_point_type == int(gpudrive.EntityType.RoadLine)
                        or road_point_type == int(gpudrive.EntityType.RoadLane)
                    ):
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

                            # Plot the road edge as a line
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

                    elif road_point_type == int(gpudrive.EntityType.SpeedBump):
                        utils.plot_speed_bumps(
                            x_coords,
                            y_coords,
                            segment_lengths,
                            segment_widths,
                            segment_orientations,
                            ax,
                        )

                    elif road_point_type == int(gpudrive.EntityType.StopSign):
                        for x, y in zip(x_coords, y_coords):
                            point = np.array([x, y])
                            utils.plot_stop_sign(
                                point=point,
                                ax=ax,
                                radius=1.5,
                                facecolor="#c04000",
                                edgecolor="none",
                                linewidth=3.0,
                                alpha=0.9,
                            )
                    elif road_point_type == int(gpudrive.EntityType.CrossWalk):
                        for x, y, length, width, orientation in zip(
                            x_coords,
                            y_coords,
                            segment_lengths,
                            segment_widths,
                            segment_orientations,
                        ):
                            points = self._get_corners_polygon(
                                x, y, length, width, orientation
                            )
                            utils.plot_crosswalk(
                                points=points,
                                ax=ax,
                                facecolor="none",
                                edgecolor="xkcd:bluish grey",
                                alpha=0.4,
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
        ) & ~self.controlled_agent_mask[env_idx, :]

        pos_x = agent_states.pos_x[env_idx, log_replay]
        pos_y = agent_states.pos_y[env_idx, log_replay]
        rotation_angle = agent_states.rotation_angle[env_idx, log_replay]
        vehicle_length = agent_states.vehicle_length[env_idx, log_replay]
        vehicle_width = agent_states.vehicle_width[env_idx, log_replay]

        # Define realistic bounds for log_replay agent positions
        valid_mask = (
            (torch.abs(pos_x) < OUT_OF_BOUNDS)
            & (torch.abs(pos_y) < OUT_OF_BOUNDS)
            & (
                (vehicle_length > 0.5)
                & (vehicle_length < 15)
                & (vehicle_width > 0.5)
                & (vehicle_width < 15)
            )
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

    def _plot_expert_trajectories(
        self,
        ax: matplotlib.axes.Axes,
        env_idx: int,
        expert_trajectories: torch.Tensor,
        response_type: Any,
    ) -> None:
        """Plot expert trajectories.
        Args:
            ax: Matplotlib axis for plotting.
            env_idx: Environment index to select specific environment agents.
            expert_trajectories: The global state of expert from `LogTrajectory`.
        """
        if self.vis_config.draw_expert_trajectories:
            controlled_mask = self.controlled_agents[env_idx, :]
            non_controlled_mask = ~response_type.static[env_idx, :] & response_type.moving[env_idx, :] & ~controlled_mask
            mask = (
                controlled_mask
                if self.vis_config.draw_only_controllable_veh
                else controlled_mask | non_controlled_mask
            )
            agent_indices = torch.where(mask)[0]
            trajectories = expert_trajectories[env_idx][mask]
            for idx, trajectory in zip(agent_indices, trajectories):
                color = AGENT_COLOR_BY_STATE["ok"] if controlled_mask[idx] else AGENT_COLOR_BY_STATE["log_replay"]
                for step in trajectory:
                    x, y = step[:2].numpy()
                    if x < OUT_OF_BOUNDS and y < OUT_OF_BOUNDS:
                        ax.add_patch(
                            Circle(
                                (x, y),
                                radius=0.3,
                                color=color,
                                fill=True,
                                alpha=0.5,
                            )
                        )

    def plot_agent_observation(
        self,
        agent_idx: int,
        env_idx: int,
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

        # Plot roadgraph if provided
        if observation_roadgraph is not None:
            for road_type, type_name in ROAD_GRAPH_TYPE_NAMES.items():
                mask = (
                    observation_roadgraph.type[env_idx, agent_idx, :]
                    == road_type
                )

                # Extract relevant roadgraph data for plotting
                x_points = observation_roadgraph.x[env_idx, agent_idx, mask]
                y_points = observation_roadgraph.y[env_idx, agent_idx, mask]
                orientations = observation_roadgraph.orientation[
                    env_idx, agent_idx, mask
                ]
                segment_lengths = observation_roadgraph.segment_length[
                    env_idx, agent_idx, mask
                ]
                widths = observation_roadgraph.segment_width[
                    env_idx, agent_idx, mask
                ]

                # Scatter plot for the points
                ax.scatter(
                    x_points,
                    y_points,
                    c=[ROAD_GRAPH_COLORS[road_type]],
                    s=8,
                    label=type_name,
                )

                # Plot lines for road edges
                for x, y, orientation, segment_length, width in zip(
                    x_points, y_points, orientations, segment_lengths, widths
                ):
                    dx = segment_length * 0.5 * np.cos(orientation)
                    dy = segment_length * 0.5 * np.sin(orientation)

                    # Calculate line endpoints for the road edge
                    x_start = x - dx
                    y_start = y - dy
                    x_end = x + dx
                    y_end = y + dy

                    # Add width as a perpendicular offset
                    width_dx = width * 0.5 * np.sin(orientation)
                    width_dy = -width * 0.5 * np.cos(orientation)

                    # Draw the road edge as a polygon (line with width)
                    ax.plot(
                        [x_start - width_dx, x_end - width_dx],
                        [y_start - width_dy, y_end - width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=1.0,
                    )
                    ax.plot(
                        [x_start + width_dx, x_end + width_dx],
                        [y_start + width_dy, y_end + width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=1.0,
                    )
                    ax.plot(
                        [x_start - width_dx, x_start + width_dx],
                        [y_start - width_dy, y_start + width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=1.0,
                    )
                    ax.plot(
                        [x_end - width_dx, x_end + width_dx],
                        [y_end - width_dy, y_end + width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=1.0,
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
                alpha=1.0,
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
                head_width=1.0,
                head_length=1.1,
                fc=REL_OBS_OBJ_COLORS["ego"],
                ec=REL_OBS_OBJ_COLORS["ego"],
            )

            ax.scatter(
                observation_ego.rel_goal_x[env_idx, agent_idx],
                observation_ego.rel_goal_y[env_idx, agent_idx],
                s=5,
                linewidth=1.5,
                c=ego_agent_color,
                marker="x",
            )

            circle = Circle(
                (
                    observation_ego.rel_goal_x[env_idx, agent_idx],
                    observation_ego.rel_goal_y[env_idx, agent_idx],
                ),
                radius=self.goal_radius,
                color=ego_agent_color,
                fill=False,
                linestyle="--",
            )
            ax.add_patch(circle)

            observation_radius = Circle(
                (0, 0),
                radius=self.env_config.obs_radius,
                color="#000000",
                linewidth=0.8,
                fill=False,
                linestyle="-",
            )
            ax.add_patch(observation_radius)
            plt.axis("off")

        ax.set_xlim((-self.env_config.obs_radius, self.env_config.obs_radius))
        ax.set_ylim((-self.env_config.obs_radius, self.env_config.obs_radius))
        ax.set_xticks([])
        ax.set_yticks([])

        return fig
