import torch
import matplotlib

matplotlib.use("Agg")
from typing import Tuple, Optional, List, Dict, Any, Union
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import ListedColormap
from jaxlib.xla_extension import ArrayImpl
import numpy as np
import madrona_gpudrive
from gpudrive.visualize import utils
from gpudrive.datatypes.roadgraph import (
    LocalRoadGraphPoints,
    GlobalRoadGraphPoints,
)
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
)
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.control import ResponseType
from gpudrive.visualize.color import (
    ROAD_GRAPH_COLORS,
    ROAD_GRAPH_TYPE_NAMES,
    REL_OBS_OBJ_COLORS,
    AGENT_COLOR_BY_STATE,
    AGENT_COLOR_BY_POLICY,
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
        self.backend = "torch"  # Always use torch or np for plotting
        self.device = "cpu"
        self.goal_radius = goal_radius
        self.num_worlds = num_worlds
        self.render_config = render_config
        self.figsize = (15, 15)
        self.env_config = env_config
        self.render_3d = render_config.render_3d
        self.vehicle_height = (
            render_config.vehicle_height
        )  # Default vehicle height
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
        self.controlled_agent_mask = controlled_agent_mask

        if isinstance(controlled_agent_mask, ArrayImpl):
            self.controlled_agent_mask = torch.from_numpy(
                np.array(controlled_agent_mask)
            )

        self.controlled_agent_mask = self.controlled_agent_mask.to(self.device)

        self.log_trajectory = LogTrajectory.from_tensor(
            self.sim_object.expert_trajectory_tensor(),
            self.num_worlds,
            self.controlled_agent_mask.shape[1],
            backend=self.backend,
        )

    def plot_simulator_state(
        self,
        env_indices: List[int],
        time_steps: Optional[List[int]] = None,
        center_agent_indices: Optional[List[int]] = None,
        zoom_radius: int = 100,
        plot_log_replay_trajectory: bool = False,
        agent_positions: Optional[torch.Tensor] = None,
        backward_goals: bool = False,
        policy_masks: Optional[Dict[int,Dict[str,torch.Tensor]]] = None,
    ):
        """
        Plot simulator states for one or multiple environments.

        Args:
            env_indices: List of environment indices to plot.
            time_steps: Optional list of time steps corresponding to each environment.
            center_agent_indices: Optional list of center agent indices for zooming.
            figsize: Tuple for figure size of each subplot.
            zoom_radius: Radius for zooming in around the center agent.
            plot_log_replay_trajectory: If True, plots the log replay trajectory.
            agent_positions: Optional tensor to plot rolled out agent positions.
            backward_goals: If True, plots backward goals for controlled agents.
            policy_mask: dict
            A dictionary that maps policies to world and specifies which agents are assigned to each policy. 
            For now maximum number of policies is 3 as there are only 3 colors in COLOR_AGENT_BY_POLICY
            The structure follows the format: {Policy Name: (Policy Function,mask) }, where:
                - Policy (str): The policy assigned to agents within the world.

                - Policy Function  (Neural Network): The identifier for the simulation environment.
                
                - Mask (torch.Tensor): A boolean or index-based mask indicating which agents follow the given policy, for all worlds. 
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

        if backward_goals:

            # Get world means for coordinate transformation
            means_xy = (
                self.sim_object.world_means_tensor()
                .to_torch()[:, :2]
                .to(self.device)
            )

            # Create extended goals dictionary
            extended_goals = {
                "x": torch.zeros_like(global_agent_states.goal_x),
                "y": torch.zeros_like(global_agent_states.goal_y),
            }
            # Generate reverse offsets for controlled agents
            for env_idx in env_indices:
                controlled_mask = self.controlled_agent_mask[env_idx]

                # Calculate direction vectors for each agent (from initial position to original goal)
                direction_x = (
                    global_agent_states.goal_x[env_idx]
                    - global_agent_states.pos_x[env_idx]
                )
                direction_y = (
                    global_agent_states.goal_y[env_idx]
                    - global_agent_states.pos_y[env_idx]
                )

                # Store extended goals - place them in opposite direction from current position
                # For controlled agents, the new goal will be behind them relative to their original goal
                extended_goals["x"][env_idx] = (
                    global_agent_states.pos_x[env_idx] - direction_x
                )
                extended_goals["y"][env_idx] = (
                    global_agent_states.pos_y[env_idx] - direction_y
                )

                # Only modify goals for controlled agents
                uncontrolled_mask = ~controlled_mask
                extended_goals["x"][
                    env_idx, uncontrolled_mask
                ] = global_agent_states.goal_x[env_idx, uncontrolled_mask]
                extended_goals["y"][
                    env_idx, uncontrolled_mask
                ] = global_agent_states.goal_y[env_idx, uncontrolled_mask]

                # Print information for controlled agents
                for agent_idx in torch.where(controlled_mask)[0]:
                    # Get original goal in world coordinates
                    orig_goal_x = (
                        global_agent_states.goal_x[env_idx, agent_idx]
                        + means_xy[env_idx, 0]
                    )
                    orig_goal_y = (
                        global_agent_states.goal_y[env_idx, agent_idx]
                        + means_xy[env_idx, 1]
                    )

                    # Get extended goal in world coordinates
                    ext_goal_x = (
                        extended_goals["x"][env_idx, agent_idx]
                        + means_xy[env_idx, 0]
                    )
                    ext_goal_y = (
                        extended_goals["y"][env_idx, agent_idx]
                        + means_xy[env_idx, 1]
                    )

                    print(
                        f"Agent ID: {global_agent_states.id[env_idx, agent_idx].item()}"
                    )
                    print(
                        f"Original goal (world coords): ({orig_goal_x.item():.6f}, {orig_goal_y.item():.6f})"
                    )
                    print(
                        f"Extended goal (world coords): ({ext_goal_x.item():.6f}, {ext_goal_y.item():.6f})"
                    )
                    print(
                        f"World mean: ({means_xy[env_idx, 0].item():.6f}, {means_xy[env_idx, 1].item():.6f})\n"
                    )

        else:
            extended_goals = None

        agent_infos = (
            self.sim_object.info_tensor().to_torch().clone().to(self.device)
        )

        figs = []

        # Calculate scale factors based on figure size
        marker_scale = max(self.figsize) / 15
        line_width_scale = max(self.figsize) / 15


        if policy_masks:

            world_based_policy_mask = {}
            
            for policy_name, (fn,mask) in policy_masks.items():
                for world in range(mask.shape[0]):
                    if world not in world_based_policy_mask:
                        world_based_policy_mask[world] = {}
                    world_based_policy_mask[world][policy_name] = mask[world]                   

        else:
            world_based_policy_mask = None

        # Iterate over each environment index
        for idx, (env_idx, time_step, center_agent_idx) in enumerate(
            zip(env_indices, time_steps, center_agent_indices)
        ):

            # Initialize figure and axes from cached road graph
            fig, ax = plt.subplots(
                figsize=self.figsize,
                subplot_kw={"projection": "3d"} if self.render_3d else {},
            )
            if self.render_3d:
                ax.view_init(elev=30, azim=45)  # Set default 3D view angle
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.clear()  # Clear any existing content
            ax.set_aspect("equal", adjustable="box")
            figs.append(fig)  # Add the new figure
            plt.close(fig)  # Close the figure to prevent carryover

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
                extended_goals=extended_goals,
                world_based_policy_mask=world_based_policy_mask,
            )

            if agent_positions is not None:
                # First calculate the maximum valid trajectory length across all agents for this env_idx
                max_valid_length = 0
                for agent_idx in range(agent_positions.shape[1]):
                    if controlled_live[agent_idx]:
                        trajectory = agent_positions[
                            env_idx, agent_idx, :time_step, :
                        ]
                        valid_mask = (
                            (trajectory[:, 0] != 0)
                            & (trajectory[:, 1] != 0)
                            & (torch.abs(trajectory[:, 0]) < OUT_OF_BOUNDS)
                            & (torch.abs(trajectory[:, 1]) < OUT_OF_BOUNDS)
                        )
                        max_valid_length = max(
                            max_valid_length, valid_mask.sum().item()
                        )

                # Create color palette
                palette = sns.light_palette(AGENT_COLOR_BY_STATE["ok"])
                cmap = ListedColormap(palette)
                norm = plt.Normalize(vmin=0, vmax=max_valid_length)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

                for agent_idx in range(agent_positions.shape[1]):
                    if controlled_live[agent_idx]:
                        trajectory = agent_positions[
                            env_idx, agent_idx, :time_step, :
                        ]
                        valid_mask = (
                            (trajectory[:, 0] != 0)
                            & (trajectory[:, 1] != 0)
                            & (torch.abs(trajectory[:, 0]) < OUT_OF_BOUNDS)
                            & (torch.abs(trajectory[:, 1]) < OUT_OF_BOUNDS)
                        )
                        # Get valid trajectory points
                        valid_trajectory = trajectory[valid_mask]

                        if len(valid_trajectory) > 1:
                            points = valid_trajectory.cpu().numpy()

                            if self.render_3d:
                                trajectory_height = 0.05
                                segments_3d = []
                                for i in range(len(points) - 1):
                                    segment = np.array(
                                        [
                                            [
                                                points[i][0],
                                                points[i][1],
                                                trajectory_height,
                                            ],
                                            [
                                                points[i + 1][0],
                                                points[i + 1][1],
                                                trajectory_height,
                                            ],
                                        ]
                                    )
                                    segments_3d.append(segment)

                                # Adjust color mapping to use actual position in the valid trajectory
                                t = np.linspace(
                                    0, len(segments_3d), len(segments_3d)
                                )
                                colors = cmap(norm(t))
                                colors[:, 3] = np.linspace(
                                    0.3, 0.9, len(segments_3d)
                                )

                                lc = Line3DCollection(
                                    segments_3d,
                                    colors=colors,
                                    linewidth=5,
                                    zorder=1,
                                )
                                ax.add_collection3d(lc)
                            else:
                                segments = []
                                for i in range(len(points) - 1):
                                    segment = np.array(
                                        [
                                            [points[i][0], points[i][1]],
                                            [
                                                points[i + 1][0],
                                                points[i + 1][1],
                                            ],
                                        ]
                                    )
                                    segments.append(segment)

                                # Adjust color mapping to use actual position in the valid trajectory
                                t = np.linspace(
                                    0, len(segments), len(segments)
                                )
                                colors = cmap(norm(t))
                                colors[:, 3] = np.linspace(
                                    0.3, 0.9, len(segments)
                                )

                                lc = LineCollection(
                                    segments,
                                    colors=colors,
                                    linewidth=5,
                                    zorder=1,
                                )
                                ax.add_collection(lc)

                # Add the colorbar
                try:
                    fig = ax.get_figure()
                    cbar_ax = fig.add_axes([0.92, 0.09, 0.02, 0.8])
                    cbar = fig.colorbar(sm, cax=cbar_ax)
                    cbar.set_label("Timestep", fontsize=15 * marker_scale)
                    cbar.ax.tick_params(labelsize=12 * marker_scale)
                except Exception as e:
                    print(f"Warning: Could not add colorbar: {e}")

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

            # 3d plot settings
            if self.render_3d:
                ax.set_zlim(0, zoom_radius * 0.05)
                ax.set_zticks([])
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

            ax.set_axis_off()

        for fig in figs:
            fig.tight_layout(pad=2, rect=[0.00, 0.00, 0.9, 1])

        return figs

    def _plot_log_replay_trajectory(
        self,
        ax: matplotlib.axes.Axes,
        env_idx: int,
        control_mask: torch.Tensor,
        log_trajectory: LogTrajectory,
        line_width_scale: int = 1.0,
    ):
        """Plot the log replay trajectory for controlled agents in either 2D or 3D."""
        if self.render_3d:
            # Get trajectory points
            trajectory_points = log_trajectory.pos_xy[
                env_idx, control_mask, :, :
            ].numpy()

            # Set a fixed height for trajectory visualization
            trajectory_height = 0.05  # Small height above ground

            # Plot trajectories for each controlled agent
            for agent_trajectory in trajectory_points:
                # Filter out invalid points (zeros or out of bounds)
                valid_mask = (
                    (agent_trajectory[:, 0] != 0)
                    & (agent_trajectory[:, 1] != 0)
                    & (np.abs(agent_trajectory[:, 0]) < OUT_OF_BOUNDS)
                    & (np.abs(agent_trajectory[:, 1]) < OUT_OF_BOUNDS)
                )
                valid_points = agent_trajectory[valid_mask]

                if len(valid_points) > 1:
                    # Create segments for the trajectory
                    segments = []
                    for i in range(len(valid_points) - 1):
                        segment = np.array(
                            [
                                [
                                    valid_points[i, 0],
                                    valid_points[i, 1],
                                    trajectory_height,
                                ],
                                [
                                    valid_points[i + 1, 0],
                                    valid_points[i + 1, 1],
                                    trajectory_height,
                                ],
                            ]
                        )
                        segments.append(segment)

                    # Create line collection with fade effect
                    colors = np.zeros((len(segments), 4))
                    colors[:, 1] = 0.9  # Green component
                    colors[:, 3] = np.linspace(
                        0.2, 0.6, len(segments)
                    )  # Alpha gradient

                    lc = Line3DCollection(
                        segments, colors=colors, linewidth=2 * line_width_scale
                    )
                    ax.add_collection3d(lc)

                    # Add points at trajectory positions
                    ax.scatter3D(
                        valid_points[:, 0],
                        valid_points[:, 1],
                        np.full_like(valid_points[:, 0], trajectory_height),
                        color="lightgreen",
                        s=10,
                        alpha=0.5,
                        zorder=0,
                    )
        else:
            # Original 2D plotting
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

    def _plot_3d_road_segment(
        self, ax, start, end, width, height, color, line_width=1.0
    ):
        """Helper method to plot 3D road segment with width and height."""
        # Calculate direction vector
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        length = np.linalg.norm(direction)
        if length == 0:
            return

        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])

        # Create vertices for 3D box
        vertices = []
        for z in [0, height]:  # Bottom and top faces
            vertices.extend(
                [
                    [
                        start[0] - perpendicular[0] * width / 2,
                        start[1] - perpendicular[1] * width / 2,
                        z,
                    ],
                    [
                        start[0] + perpendicular[0] * width / 2,
                        start[1] + perpendicular[1] * width / 2,
                        z,
                    ],
                    [
                        end[0] + perpendicular[0] * width / 2,
                        end[1] + perpendicular[1] * width / 2,
                        z,
                    ],
                    [
                        end[0] - perpendicular[0] * width / 2,
                        end[1] - perpendicular[1] * width / 2,
                        z,
                    ],
                ]
            )

        # Create faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side 3
            [vertices[3], vertices[0], vertices[4], vertices[7]],  # Side 4
        ]

        # Create 3D collection and add to axis
        poly3d = Poly3DCollection(faces, alpha=0.7)
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)

    def _plot_3d_polygon(
        self, ax, points, height, facecolor, edgecolor=None, alpha=1.0
    ):
        """Helper method to plot 3D polygon with height."""
        points = np.array(points)
        vertices = []

        # Create bottom and top faces
        for z in [0, height]:
            for point in points:
                vertices.append([point[0], point[1], z])

        vertices = np.array(vertices)
        n_points = len(points)

        # Create faces
        faces = []
        # Bottom face
        faces.append(vertices[:n_points])
        # Top face
        faces.append(vertices[n_points:])
        # Side faces
        for i in range(n_points):
            next_i = (i + 1) % n_points
            faces.append(
                [
                    vertices[i],
                    vertices[next_i],
                    vertices[next_i + n_points],
                    vertices[i + n_points],
                ]
            )

        # Create 3D collection and add to axis
        poly3d = Poly3DCollection(faces, alpha=alpha, zorder=1)
        poly3d.set_facecolor(facecolor)
        if edgecolor:
            poly3d.set_edgecolor(edgecolor)
        ax.add_collection3d(poly3d)

    def _plot_3d_stop_sign(
        self, ax, x, y, radius, height, facecolor, alpha=1.0
    ):
        """Helper method to plot 3D stop sign."""
        # Create octagon points
        n_sides = 8
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        points = [
            [x + radius * np.cos(angle), y + radius * np.sin(angle)]
            for angle in angles
        ]

        # Plot as 3D polygon
        self._plot_3d_polygon(ax, points, height, facecolor, alpha=alpha)

        # Add pole
        pole_radius = radius * 0.1
        pole_points = [
            [x - pole_radius, y - pole_radius],
            [x + pole_radius, y - pole_radius],
            [x + pole_radius, y + pole_radius],
            [x - pole_radius, y + pole_radius],
        ]
        self._plot_3d_polygon(
            ax, pole_points, height * 0.8, facecolor="#808080", alpha=alpha
        )

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
            if road_point_type == int(madrona_gpudrive.EntityType._None):
                continue

            road_mask = road_graph.type[env_idx, :] == road_point_type

            # Get coordinates and metadata for the current road type
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

            if road_point_type in [
                int(madrona_gpudrive.EntityType.RoadEdge),
                int(madrona_gpudrive.EntityType.RoadLine),
                int(madrona_gpudrive.EntityType.RoadLane),
            ]:
                # Handle road edges, lines, and lanes
                if self.render_3d:
                    for x, y, length, width, orientation in zip(
                        x_coords,
                        y_coords,
                        segment_lengths,
                        segment_widths,
                        segment_orientations,
                    ):
                        start, end = self._get_endpoints(
                            x, y, length, orientation
                        )

                        # Create 3D road segment
                        if road_point_type == int(
                            madrona_gpudrive.EntityType.RoadEdge
                        ):
                            # For road edges, create raised borders
                            height = 0.01  # Small height for road edges
                            self._plot_3d_road_segment(
                                ax,
                                start,
                                end,
                                width,
                                height,
                                ROAD_GRAPH_COLORS[road_point_type],
                                line_width=1.1 * line_width_scale,
                            )
                        else:
                            # For lanes and lines, plot at ground level
                            ax.plot3D(
                                [start[0], end[0]],
                                [start[1], end[1]],
                                [0, 0],  # Ground level
                                color=ROAD_GRAPH_COLORS[road_point_type],
                                linewidth=1.25 * line_width_scale,
                            )
                else:
                    # Original 2D plotting
                    for x, y, length, orientation in zip(
                        x_coords,
                        y_coords,
                        segment_lengths,
                        segment_orientations,
                    ):
                        start, end = self._get_endpoints(
                            x, y, length, orientation
                        )
                        line_width = (
                            1.1 * line_width_scale
                            if road_point_type
                            == int(madrona_gpudrive.EntityType.RoadEdge)
                            else 0.75 * line_width_scale
                        )

                        ax.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            color=ROAD_GRAPH_COLORS[road_point_type],
                            linewidth=line_width,
                        )

            elif road_point_type == int(madrona_gpudrive.EntityType.SpeedBump):
                if self.render_3d:
                    for x, y, length, width, orientation in zip(
                        x_coords,
                        y_coords,
                        segment_lengths,
                        segment_widths,
                        segment_orientations,
                    ):
                        # Create 3D speed bump with height
                        points = self._get_corners_polygon(
                            x, y, length, width, orientation
                        )
                        height = 0.0  # Height of speed bump
                        self._plot_3d_polygon(
                            ax,
                            points,
                            height,
                            facecolor=ROAD_GRAPH_COLORS[road_point_type],
                            alpha=0.6,
                        )
                else:
                    utils.plot_speed_bumps(
                        x_coords,
                        y_coords,
                        segment_lengths,
                        segment_widths,
                        segment_orientations,
                        ax,
                    )

            elif road_point_type == int(madrona_gpudrive.EntityType.StopSign):
                if self.render_3d:
                    for x, y in zip(x_coords, y_coords):
                        # Create 3D stop sign
                        height = 0.1  # Standard stop sign height
                        radius = 0.3
                        self._plot_3d_stop_sign(
                            ax,
                            x,
                            y,
                            radius,
                            height,
                            facecolor="#c04000",
                            alpha=0.9,
                        )
                else:
                    for x, y in zip(x_coords, y_coords):
                        utils.plot_stop_sign(
                            point=np.array([x, y]),
                            ax=ax,
                            radius=1.5,
                            facecolor="#c04000",
                            edgecolor="none",
                            linewidth=3.0,
                            alpha=0.9,
                        )

            elif road_point_type == int(madrona_gpudrive.EntityType.CrossWalk):
                if self.render_3d:
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
                        height = 0.0  # Slight elevation for crosswalk
                        self._plot_3d_polygon(
                            ax,
                            points,
                            height,
                            facecolor="white",
                            edgecolor="xkcd:bluish grey",
                            alpha=0.4,
                        )
                else:
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
                # Handle other road point types as markers
                if self.render_3d:
                    ax.scatter3D(
                        road_graph.x[env_idx, road_mask],
                        road_graph.y[env_idx, road_mask],
                        np.zeros_like(
                            road_graph.x[env_idx, road_mask]
                        ),  # Ground level
                        s=5 * marker_size_scale,
                        label=road_point_type,
                        color=ROAD_GRAPH_COLORS[int(road_point_type)],
                    )
                else:
                    ax.scatter(
                        road_graph.x[env_idx, road_mask],
                        road_graph.y[env_idx, road_mask],
                        s=5 * marker_size_scale,
                        label=road_point_type,
                        color=ROAD_GRAPH_COLORS[int(road_point_type)],
                    )

    def _create_3d_vehicle_box(self, x, y, length, width, orientation):
        """Create simple 3D cuboid vertices and faces for vehicle representation."""
        # Rotation matrix
        c, s = np.cos(orientation), np.sin(orientation)
        R = np.array([[c, -s], [s, c]])

        # Define base points for cuboid
        base_points = np.array(
            [
                [-length / 2, -width / 2],  # Back left
                [length / 2, -width / 2],  # Front left
                [length / 2, width / 2],  # Front right
                [-length / 2, width / 2],  # Back right
            ]
        )

        # Rotate and translate points
        transformed_points = base_points @ R.T + np.array([x, y])

        # Create 3D points
        bottom = np.column_stack(
            [transformed_points, np.zeros_like(transformed_points[:, 0])]
        )
        top = np.column_stack(
            [
                transformed_points,
                np.full_like(transformed_points[:, 0], self.vehicle_height),
            ]
        )

        # Define faces (6 faces for cuboid)
        faces = [
            bottom,  # Bottom face
            top,  # Top face
            np.array([bottom[0], bottom[1], top[1], top[0]]),  # Left side
            np.array([bottom[1], bottom[2], top[2], top[1]]),  # Front
            np.array([bottom[2], bottom[3], top[3], top[2]]),  # Right side
            np.array([bottom[3], bottom[0], top[0], top[3]]),  # Back
        ]

        return faces

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
        extended_goals: Optional[Dict[str, torch.Tensor]] = None,
        world_based_policy_mask : Optional[Dict[int,Dict[str,torch.Tensor]]] = None,
    ) -> None:
        """Plots bounding boxes for agents filtered by environment index and mask.

        Args:
            env_idx: Environment indices to select specific environments.
            ax: Matplotlib axis for plotting.
            agent_state: The global state of agents from `GlobalEgoState`.
            is_ok_mask: Mask for agents that are in a valid state.
            is_offroad_mask: Mask for agents that are off-road.
            is_collided_mask: Mask for agents that have collided.
            response_type: Mask to filter static agents.
            alpha: Alpha value for drawing, i.e., 0 means fully transparent.
            as_center_pts: If True, only plot center points instead of full boxes.
            label: Label for the plotted elements.
            plot_goal_points: If True, plot goal points for agents.
            line_width_scale: Scale factor for line width.
            marker_size_scale: Scale factor for marker size.
            extended_goals: Optional dictionary of backward goals for controlled agents.
        """

        def plot_agent_group_3d(
            bboxes, color, alpha=1.0, line_width_scale=1.5
        ):
            """Helper function to plot a group of agents in 3D"""
            for x, y, length, width, angle in bboxes:
                # Create 3D vehicle box
                faces = self._create_3d_vehicle_box(x, y, length, width, angle)

                # Plot the cuboid (vehicle box)
                poly3d = Poly3DCollection(
                    faces, alpha=alpha, zsort="max", zorder=6
                )
                poly3d.set_facecolor(color)
                poly3d.set_edgecolor("black")
                poly3d.set_linewidth(0.5 * line_width_scale)
                ax.add_collection3d(poly3d)

                # Heading arrow (use a small 3D line to indicate the orientation)
                c = np.cos(angle)
                s = np.sin(angle)
                arrow_length = 4.5

                # Coordinates of the arrow's base (center of the box) and the tip
                arrow_base = np.array(
                    [x, y, 0]
                )  # Starting point (at the top of the box)
                arrow_tip = arrow_base + np.array(
                    [arrow_length * c, arrow_length * s, 0]
                )  # Pointing in the direction of the angle

                # Plot the heading arrow
                ax.plot(
                    [arrow_base[0], arrow_tip[0]],
                    [arrow_base[1], arrow_tip[1]],
                    [arrow_base[2], arrow_tip[2]],
                    color="black",
                    linewidth=2,
                    alpha=alpha,
                    zorder=5,
                )

                # Add arrowhead (tip)
                tip_angle = np.pi / 1.5  # Angle of the arrowhead
                arrowhead_length = arrow_length / 8  # Length of the arrowhead

                # Calculate the left and right arrowhead points
                arrowhead_left = arrow_tip + np.array(
                    [
                        arrowhead_length * (np.cos(angle + tip_angle) - c),
                        arrowhead_length * (np.sin(angle + tip_angle) - s),
                        0,
                    ]
                )
                arrowhead_right = arrow_tip + np.array(
                    [
                        arrowhead_length * (np.cos(angle - tip_angle) - c),
                        arrowhead_length * (np.sin(angle - tip_angle) - s),
                        0,
                    ]
                )

                # Plot the left and right arrowhead lines
                ax.plot(
                    [arrow_tip[0], arrowhead_left[0]],
                    [arrow_tip[1], arrowhead_left[1]],
                    [arrow_tip[2], arrowhead_left[2]],
                    color="black",
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=5,
                )
                ax.plot(
                    [arrow_tip[0], arrowhead_right[0]],
                    [arrow_tip[1], arrowhead_right[1]],
                    [arrow_tip[2], arrowhead_right[2]],
                    color="black",
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=5,
                )


        def plot_agent_group_2d(bboxes, color,by_policy = False):
            """Helper function to plot a group of agents in 2D"""
            if not by_policy:
                utils.plot_numpy_bounding_boxes(
                    ax=ax,
                    bboxes=bboxes,
                    color=color,
                    alpha=alpha,
                    line_width_scale=line_width_scale,
                    as_center_pts=as_center_pts,
                    label=label,
                )
            else:
                num_policies = len(bboxes)
                utils.plot_numpy_bounding_boxes_multiple_policy(            
                ax=ax,
                bboxes_s=bboxes,
                colors=color[:num_policies],
                alpha=alpha,
                line_width_scale=line_width_scale,
                as_center_pts=as_center_pts,
                label=label,
                    )
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

        if self.render_3d:
            plot_agent_group_3d(
                bboxes_controlled_offroad, AGENT_COLOR_BY_STATE["off_road"]
            )
        else:
            plot_agent_group_2d(
                bboxes_controlled_offroad, AGENT_COLOR_BY_STATE["off_road"]
            )

        # Plot goals
        if plot_goal_points:
            for mask, color in [
                (is_ok_mask, AGENT_COLOR_BY_STATE["ok"]),
                (is_offroad_mask, AGENT_COLOR_BY_STATE["off_road"]),
                (is_collided_mask, AGENT_COLOR_BY_STATE["collided"]),
            ]:
                if not mask.any():
                    continue

                goal_x = agent_states.goal_x[env_idx, mask].numpy()
                goal_y = agent_states.goal_y[env_idx, mask].numpy()

                if self.render_3d:
                    # Plot goals as vertical lines in 3D
                    for x, y in zip(goal_x, goal_y):
                        ax.plot3D(
                            [x, x],
                            [y, y],
                            [0, self.vehicle_height],
                            color=color,
                            linestyle="--",
                            linewidth=2 * line_width_scale,
                        )
                        # Add goal circle on the ground
                        circle_points = np.linspace(0, 2 * np.pi, 32)
                        circle_x = x + self.goal_radius * np.cos(circle_points)
                        circle_y = y + self.goal_radius * np.sin(circle_points)
                        circle_z = np.zeros_like(circle_points)
                        ax.plot3D(
                            circle_x,
                            circle_y,
                            circle_z,
                            color=color,
                            linestyle="--",
                            linewidth=2 * line_width_scale,
                        )
                else:
                    # Original 2D goal plotting
                    ax.scatter(
                        goal_x,
                        goal_y,
                        s=5 * marker_size_scale,
                        linewidth=1.5 * line_width_scale,
                        c=color,
                        marker="o",
                    )
                    for x, y in zip(goal_x, goal_y):
                        circle = Circle(
                            (x, y),
                            radius=self.goal_radius,
                            color=color,
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

        if self.render_3d:
            plot_agent_group_3d(
                bboxes_controlled_collided, AGENT_COLOR_BY_STATE["collided"]
            )
        else:
            plot_agent_group_2d(
                bboxes_controlled_collided, AGENT_COLOR_BY_STATE["collided"]
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

        if not world_based_policy_mask: ## controlled by the same policy
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
            if self.render_3d:
                plot_agent_group_3d(
                    bboxes_controlled_ok, AGENT_COLOR_BY_STATE["ok"]
                )
            else:
                plot_agent_group_2d(
                    bboxes_controlled_ok, AGENT_COLOR_BY_STATE["ok"]
                )

        else:
            bboxes_controlled_ok = []
            policy_mask = world_based_policy_mask[env_idx]
            for policy_name,mask in policy_mask.items():

                bboxes = np.stack(
                    (
                        agent_states.pos_x[env_idx, mask].numpy(),
                        agent_states.pos_y[env_idx, mask].numpy(),
                        agent_states.vehicle_length[env_idx, mask].numpy(),
                        agent_states.vehicle_width[env_idx, mask].numpy(),
                        agent_states.rotation_angle[env_idx, mask].numpy(),
                    ),
                    axis=1,
                    )
                bboxes_controlled_ok.append(bboxes)

            plot_agent_group_2d(
                bboxes_controlled_ok, AGENT_COLOR_BY_POLICY,by_policy=True
            )
        # Plot log replay agents
        log_replay = (
            response_type.static[env_idx, :] | response_type.moving[env_idx, :]
        ) & ~self.controlled_agent_mask[env_idx, :]

        pos_x = agent_states.pos_x[env_idx, log_replay]
        pos_y = agent_states.pos_y[env_idx, log_replay]
        rotation_angle = agent_states.rotation_angle[env_idx, log_replay]
        vehicle_length = agent_states.vehicle_length[env_idx, log_replay]
        vehicle_width = agent_states.vehicle_width[env_idx, log_replay]

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

        if self.render_3d:
            plot_agent_group_3d(
                bboxes_static, AGENT_COLOR_BY_STATE["log_replay"]
            )
        else:
            plot_agent_group_2d(
                bboxes_static, AGENT_COLOR_BY_STATE["log_replay"]
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
            non_controlled_mask = (
                ~response_type.static[env_idx, :]
                & response_type.moving[env_idx, :]
                & ~controlled_mask
            )
            mask = (
                controlled_mask
                if self.vis_config.draw_only_controllable_veh
                else controlled_mask | non_controlled_mask
            )
            agent_indices = torch.where(mask)[0]
            trajectories = expert_trajectories[env_idx][mask]
            for idx, trajectory in zip(agent_indices, trajectories):
                color = (
                    AGENT_COLOR_BY_STATE["ok"]
                    if controlled_mask[idx]
                    else AGENT_COLOR_BY_STATE["log_replay"]
                )
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
        trajectory: Optional[np.ndarray] = None,
    ):
        """
        Plot observation from agent POV to inspect the information available 
        to the agent.
        Args:
            agent_idx (int): Index of the agent whose observation is to be plotted.
            env_idx (int): Index of the environment in the batch.
            trajectory (Optional[np.ndarray], optional): Array of trajectory points to plot.
                Should be of shape (N, 2) where N is the number of points and each point
                is an (x, y) coordinate. Defaults to None.
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

        if trajectory is not None and len(trajectory) > 0:
            # Plot the trajectory as a line
            ax.plot(
                trajectory[:, 0],  # x coordinates
                trajectory[:, 1],  # y coordinates
                color='blue',      # trajectory color
                linestyle='-',     # solid line
                linewidth=1.0,     # line width
                marker='o',        # circular markers at each point
                markersize=1,      # size of markers
                alpha=0.7,         # slight transparency
                label='Trajectory' # label for legend
            )

        ax.set_xlim((-self.env_config.obs_radius, self.env_config.obs_radius))
        ax.set_ylim((-self.env_config.obs_radius, self.env_config.obs_radius))
        ax.set_xticks([])
        ax.set_yticks([])

        return fig
