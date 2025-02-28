import torch
import matplotlib
matplotlib.use('Agg')
from typing import Tuple, Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import pandas as pd
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
    AGENT_COLOR_BY_POLICY
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
        self.render_3d = render_config.render_3d
        self.vehicle_height = render_config.vehicle_height  # Default vehicle height
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

    def plot_simulator_state(
        self,
        env_indices: List[int],
        time_steps: Optional[List[int]] = None,
        center_agent_indices: Optional[List[int]] = None,
        zoom_radius: int = 100,
        plot_log_replay_trajectory: bool = False,
        agent_positions: Optional[torch.Tensor] = None,
        extend_goals: bool = False,
        policy_masks: dict = None,
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

        if extend_goals:
            # Initialize random number generator
            rng = np.random

            # Define the range for random goal offsets (small range centered around 0)
            OFFSET_RANGE = 8.0

            # Get world means for coordinate transformation
            means_xy = self.sim_object.world_means_tensor().to_torch()[:, :2].to(self.device)

            # Create extended goals dictionary
            extended_goals = {
                'x': torch.zeros_like(global_agent_states.goal_x),
                'y': torch.zeros_like(global_agent_states.goal_y)
            }
            # Generate random offsets for controlled agents
            for env_idx in env_indices:
                controlled_mask = self.controlled_agent_mask[env_idx]

                # Calculate direction vectors for each agent (from initial position to original goal)
                direction_x = global_agent_states.goal_x[env_idx] - global_agent_states.pos_x[env_idx]
                direction_y = global_agent_states.goal_y[env_idx] - global_agent_states.pos_y[env_idx]

                # Store extended goals - place them in opposite direction from current position
                # For controlled agents, the new goal will be behind them relative to their original goal
                extended_goals['x'][env_idx] = global_agent_states.pos_x[env_idx] - direction_x
                extended_goals['y'][env_idx] = global_agent_states.pos_y[env_idx] - direction_y

                # Only modify goals for controlled agents
                uncontrolled_mask = ~controlled_mask
                extended_goals['x'][env_idx, uncontrolled_mask] = global_agent_states.goal_x[env_idx, uncontrolled_mask]
                extended_goals['y'][env_idx, uncontrolled_mask] = global_agent_states.goal_y[env_idx, uncontrolled_mask]

                # Print information for controlled agents
                for agent_idx in torch.where(controlled_mask)[0]:
                    # Get original goal in world coordinates
                    orig_goal_x = global_agent_states.goal_x[env_idx, agent_idx] + means_xy[env_idx, 0]
                    orig_goal_y = global_agent_states.goal_y[env_idx, agent_idx] + means_xy[env_idx, 1]

                    # Get extended goal in world coordinates
                    ext_goal_x = extended_goals['x'][env_idx, agent_idx] + means_xy[env_idx, 0]
                    ext_goal_y = extended_goals['y'][env_idx, agent_idx] + means_xy[env_idx, 1]

                    print(f"Agent ID: {global_agent_states.id[env_idx, agent_idx].item()}")
                    print(f"Original goal (world coords): ({orig_goal_x.item():.6f}, {orig_goal_y.item():.6f})")
                    print(f"Extended goal (world coords): ({ext_goal_x.item():.6f}, {ext_goal_y.item():.6f})")
                    print(f"World mean: ({means_xy[env_idx, 0].item():.6f}, {means_xy[env_idx, 1].item():.6f})\n")

        else:
            extended_goals = None

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

            is_ok = ~is_offroad & ~is_collided & controlled_live ## make mask with policies

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
            if policy_masks:
                policy_mask = policy_masks[idx] 
            else:
                policy_mask = None
            
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
                policy_mask=policy_mask
            )

            if agent_positions is not None:
                for agent_idx in range(agent_positions.shape[1]):
                    if controlled_live[agent_idx]:
                        trajectory = agent_positions[env_idx, agent_idx, :time_step, :]

                        valid_mask = ((trajectory[:, 0] != 0) & (trajectory[:, 1] != 0) & 
                                    (torch.abs(trajectory[:, 0]) < OUT_OF_BOUNDS) & 
                                    (torch.abs(trajectory[:, 1]) < OUT_OF_BOUNDS))
                        
                        # Get valid trajectory points
                        valid_trajectory = trajectory[valid_mask]
                        
                        if len(valid_trajectory) > 1:
                            # Convert to numpy and ensure correct shape
                            points = valid_trajectory.cpu().numpy()
                            
                            # Create segments by pairing consecutive points
                            segments = []
                            for i in range(len(points) - 1):
                                segment = np.array([[points[i][0], points[i][1]],
                                                  [points[i+1][0], points[i+1][1]]])
                                segments.append(segment)
                            segments = np.array(segments)
                            
                            # Create color gradient
                            colors = np.zeros((len(segments), 4))  # RGBA colors
                            colors[:, 0] = np.linspace(0.114, 0.051, len(segments))  # R
                            colors[:, 1] = np.linspace(0.678, 0.278, len(segments))  # G
                            colors[:, 2] = np.linspace(0.753, 0.631, len(segments))  # B
                            colors[:, 3] = np.linspace(0.3, 0.9, len(segments))      # Alpha
                            
                            # Create line collection with color gradient
                            from matplotlib.collections import LineCollection
                            lc = LineCollection(segments, colors=colors, linewidth=5)
                            ax.add_collection(lc)
                    
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

        for fig in figs:
            fig.tight_layout(pad=0)

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
            trajectory_points = log_trajectory.pos_xy[env_idx, control_mask, :, :].numpy()
            
            # Set a fixed height for trajectory visualization
            trajectory_height = 0.05  # Small height above ground
            
            # Plot trajectories for each controlled agent
            for agent_trajectory in trajectory_points:
                # Filter out invalid points (zeros or out of bounds)
                valid_mask = ((agent_trajectory[:, 0] != 0) & 
                            (agent_trajectory[:, 1] != 0) &
                            (np.abs(agent_trajectory[:, 0]) < OUT_OF_BOUNDS) &
                            (np.abs(agent_trajectory[:, 1]) < OUT_OF_BOUNDS))
                valid_points = agent_trajectory[valid_mask]
                
                if len(valid_points) > 1:
                    # Create segments for the trajectory
                    segments = []
                    for i in range(len(valid_points) - 1):
                        segment = np.array([
                            [valid_points[i, 0], valid_points[i, 1], trajectory_height],
                            [valid_points[i+1, 0], valid_points[i+1, 1], trajectory_height]
                        ])
                        segments.append(segment)
                    
                    # Create line collection with fade effect
                    colors = np.zeros((len(segments), 4))
                    colors[:, 1] = 0.9  # Green component
                    colors[:, 3] = np.linspace(0.2, 0.6, len(segments))  # Alpha gradient
                    
                    lc = Line3DCollection(segments, colors=colors, linewidth=2 * line_width_scale)
                    ax.add_collection3d(lc)
                    
                    # Add points at trajectory positions
                    ax.scatter3D(
                        valid_points[:, 0],
                        valid_points[:, 1],
                        np.full_like(valid_points[:, 0], trajectory_height),
                        color="lightgreen",
                        s=10,
                        alpha=0.5,
                        zorder=0
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
    
    def _plot_3d_road_segment(self, ax, start, end, width, height, color, line_width=1.0):
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
            vertices.extend([
                [start[0] - perpendicular[0] * width/2, start[1] - perpendicular[1] * width/2, z],
                [start[0] + perpendicular[0] * width/2, start[1] + perpendicular[1] * width/2, z],
                [end[0] + perpendicular[0] * width/2, end[1] + perpendicular[1] * width/2, z],
                [end[0] - perpendicular[0] * width/2, end[1] - perpendicular[1] * width/2, z]
            ])
        
        # Create faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side 3
            [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side 4
        ]
        
        # Create 3D collection and add to axis
        poly3d = Poly3DCollection(faces, alpha=0.7)
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)

    def _plot_3d_polygon(self, ax, points, height, facecolor, edgecolor=None, alpha=1.0):
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
            faces.append([
                vertices[i],
                vertices[next_i],
                vertices[next_i + n_points],
                vertices[i + n_points]
            ])
        
        # Create 3D collection and add to axis
        poly3d = Poly3DCollection(faces, alpha=alpha)
        poly3d.set_facecolor(facecolor)
        if edgecolor:
            poly3d.set_edgecolor(edgecolor)
        ax.add_collection3d(poly3d)

    def _plot_3d_stop_sign(self, ax, x, y, radius, height, facecolor, alpha=1.0):
        """Helper method to plot 3D stop sign."""
        # Create octagon points
        n_sides = 8
        angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
        points = [[x + radius * np.cos(angle), y + radius * np.sin(angle)] for angle in angles]
        
        # Plot as 3D polygon
        self._plot_3d_polygon(ax, points, height, facecolor, alpha=alpha)
        
        # Add pole
        pole_radius = radius * 0.1
        pole_points = [
            [x - pole_radius, y - pole_radius],
            [x + pole_radius, y - pole_radius],
            [x + pole_radius, y + pole_radius],
            [x - pole_radius, y + pole_radius]
        ]
        self._plot_3d_polygon(ax, pole_points, height * 0.8, facecolor="#808080", alpha=alpha)

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
            segment_lengths = road_graph.segment_length[env_idx, road_mask].tolist()
            segment_widths = road_graph.segment_width[env_idx, road_mask].tolist()
            segment_orientations = road_graph.orientation[env_idx, road_mask].tolist()
            
            if road_point_type in [
                int(madrona_gpudrive.EntityType.RoadEdge),
                int(madrona_gpudrive.EntityType.RoadLine),
                int(madrona_gpudrive.EntityType.RoadLane)
            ]:
                # Handle road edges, lines, and lanes
                if self.render_3d:
                    for x, y, length, width, orientation in zip(
                        x_coords, y_coords, segment_lengths, segment_widths, segment_orientations
                    ):
                        start, end = self._get_endpoints(x, y, length, orientation)
                        
                        # Create 3D road segment
                        if road_point_type == int(madrona_gpudrive.EntityType.RoadEdge):
                            # For road edges, create raised borders
                            height = 0.01  # Small height for road edges
                            self._plot_3d_road_segment(
                                ax, start, end, width, height,
                                ROAD_GRAPH_COLORS[road_point_type],
                                line_width=1.1 * line_width_scale
                            )
                        else:
                            # For lanes and lines, plot at ground level
                            ax.plot3D(
                                [start[0], end[0]],
                                [start[1], end[1]],
                                [0, 0],  # Ground level
                                color=ROAD_GRAPH_COLORS[road_point_type],
                                linewidth=1.25 * line_width_scale
                            )
                else:
                    # Original 2D plotting
                    for x, y, length, orientation in zip(
                        x_coords, y_coords, segment_lengths, segment_orientations
                    ):
                        start, end = self._get_endpoints(x, y, length, orientation)
                        line_width = 1.1 * line_width_scale if road_point_type == int(
                            madrona_gpudrive.EntityType.RoadEdge
                        ) else 0.75 * line_width_scale
                        
                        ax.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            color=ROAD_GRAPH_COLORS[road_point_type],
                            linewidth=line_width,
                        )
                        
            elif road_point_type == int(madrona_gpudrive.EntityType.SpeedBump):
                if self.render_3d:
                    for x, y, length, width, orientation in zip(
                        x_coords, y_coords, segment_lengths, segment_widths, segment_orientations
                    ):
                        # Create 3D speed bump with height
                        points = self._get_corners_polygon(x, y, length, width, orientation)
                        height = 0.0  # Height of speed bump
                        self._plot_3d_polygon(
                            ax, points, height,
                            facecolor=ROAD_GRAPH_COLORS[road_point_type],
                            alpha=0.6
                        )
                else:
                    utils.plot_speed_bumps(
                        x_coords, y_coords, segment_lengths,
                        segment_widths, segment_orientations, ax
                    )
                    
            elif road_point_type == int(madrona_gpudrive.EntityType.StopSign):
                if self.render_3d:
                    for x, y in zip(x_coords, y_coords):
                        # Create 3D stop sign
                        height = 0.1  # Standard stop sign height
                        radius = 0.3
                        self._plot_3d_stop_sign(
                            ax, x, y, radius, height,
                            facecolor="#c04000",
                            alpha=0.9
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
                        x_coords, y_coords, segment_lengths, segment_widths, segment_orientations
                    ):
                        points = self._get_corners_polygon(x, y, length, width, orientation)
                        height = 0.0  # Slight elevation for crosswalk
                        self._plot_3d_polygon(
                            ax, points, height,
                            facecolor="white",
                            edgecolor="xkcd:bluish grey",
                            alpha=0.4
                        )
                else:
                    for x, y, length, width, orientation in zip(
                        x_coords, y_coords, segment_lengths, segment_widths, segment_orientations
                    ):
                        points = self._get_corners_polygon(x, y, length, width, orientation)
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
                        np.zeros_like(road_graph.x[env_idx, road_mask]),  # Ground level
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
        base_points = np.array([
            [-length/2, -width/2],  # Back left
            [length/2, -width/2],   # Front left
            [length/2, width/2],    # Front right
            [-length/2, width/2],   # Back right
        ])
        
        # Rotate and translate points
        transformed_points = base_points @ R.T + np.array([x, y])
        
        # Create 3D points
        bottom = np.column_stack([transformed_points, np.zeros_like(transformed_points[:, 0])])
        top = np.column_stack([transformed_points, np.full_like(transformed_points[:, 0], self.vehicle_height)])
        
        # Define faces (6 faces for cuboid)
        faces = [
            bottom,  # Bottom face
            top,     # Top face
            np.array([bottom[0], bottom[1], top[1], top[0]]),  # Left side
            np.array([bottom[1], bottom[2], top[2], top[1]]),  # Front
            np.array([bottom[2], bottom[3], top[3], top[2]]),  # Right side
            np.array([bottom[3], bottom[0], top[0], top[3]])   # Back
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
        policy_mask : torch.Tensor = None
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
            for mask, color in [(is_ok_mask, AGENT_COLOR_BY_STATE["ok"]),
                          (is_offroad_mask, AGENT_COLOR_BY_STATE["off_road"]),
                          (is_collided_mask, AGENT_COLOR_BY_STATE["collided"])]:

                if not mask.any():
                    continue
   
                # Plot original goals
                goal_x = agent_states.goal_x[env_idx, mask].numpy()
                goal_y = agent_states.goal_y[env_idx, mask].numpy()

                # Plot original goals with 'o' marker
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

                # If we have extended goals, plot them and connect with dotted lines
                if extended_goals is not None:
                    ext_x = extended_goals['x'][env_idx, mask].numpy()
                    ext_y = extended_goals['y'][env_idx, mask].numpy()

                    # Plot extended goals with 'x' marker
                    ax.scatter(
                        ext_x,
                        ext_y,
                        s=5 * marker_size_scale,
                        linewidth=1.5 * line_width_scale,
                        c="blue",
                        marker="x",
                    )

                    # Draw circles around extended goals
                    for x, y in zip(ext_x, ext_y):
                        circle = Circle(
                            (x, y),
                            radius=self.goal_radius,
                            color="blue",
                            fill=False,
                            linestyle=":"
                        )
                        ax.add_patch(circle)

                    # Connect original and extended goals with dotted lines
                    for ox, oy, nx, ny in zip(goal_x, goal_y, ext_x, ext_y):
                        ax.plot([ox, nx], [oy, ny], 
                            color=color, 
                            linestyle=":", 
                            linewidth=1.0 * line_width_scale,
                            alpha=0.7)

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


        if not policy_mask:
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
                color=AGENT_COLOR_BY_STATE["ok"],#color,#
                alpha=alpha,
                line_width_scale=line_width_scale,
                as_center_pts=as_center_pts,
                label=label,
            )
        else:
            


            bboxes = []

            for policy_name,mask in policy_mask.items():
          
                bboxes_controlled_ok = np.stack(
                    (
                        agent_states.pos_x[env_idx, mask].numpy(),
                        agent_states.pos_y[env_idx, mask].numpy(),
                        agent_states.vehicle_length[env_idx, mask].numpy(),
                        agent_states.vehicle_width[env_idx, mask].numpy(),
                        agent_states.rotation_angle[env_idx, mask].numpy(),
                    ),
                    axis=1,
                    )
                bboxes.append(bboxes_controlled_ok)

            


            utils.plot_numpy_bounding_boxes_multiple_policy(
                ax=ax,
                bboxes_s=bboxes,
                colors=AGENT_COLOR_BY_POLICY,#AGENT_COLOR_BY_STATE["ok"],
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
