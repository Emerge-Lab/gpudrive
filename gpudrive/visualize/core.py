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
    LidarObs,
)
from gpudrive.datatypes.trajectory import LogTrajectory, VBDTrajectory
from gpudrive.datatypes.control import ResponseType
from gpudrive.env import constants
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
        reference_trajectory,
        goal_radius,
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
        self.figsize = (10, 10)
        self.env_config = env_config
        self.render_3d = render_config.render_3d
        self.vehicle_height = render_config.vehicle_height
        self.initialize_static_scenario_data(
            controlled_agent_mask=controlled_agent_mask,
            reference_trajectory=reference_trajectory,
        )

    def initialize_static_scenario_data(
        self,
        controlled_agent_mask,
        reference_trajectory,
    ):
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
        self.controlled_agent_mask = controlled_agent_mask.clone().to(
            self.device
        )

        if isinstance(controlled_agent_mask, ArrayImpl):
            self.controlled_agent_mask = torch.from_numpy(
                np.array(controlled_agent_mask)
            )

        self.trajectory = reference_trajectory

    def plot_simulator_state(
        self,
        env_indices: List[int],
        time_steps: Optional[List[int]] = None,
        center_agent_indices: Optional[List[int]] = None,
        zoom_radius: int = 100,
        plot_guidance_pos_xy: bool = False,
        plot_guidance_up_to_time: bool = False,
        agent_positions: Optional[torch.Tensor] = None,
        backward_goals: bool = False,
        policy_masks: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        colorbar: bool = False,
        multiple_rollouts: bool = False,
        line_color="#1f77b4",
        line_alpha: float = 0.3,
        line_width: float = 1.0,
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Plot simulator states for one or multiple environments.

        Args:
            env_indices: List of environment indices to plot.
            time_steps: Optional list of time steps corresponding to each environment.
            center_agent_indices: Optional list of center agent indices for zooming.
            figsize: Tuple for figure size of each subplot.
            zoom_radius: Radius for zooming in around the center agent.
            plot_guidance_pos_xy: If True, plots the waypoints from the human replays.
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

        self.multi_rollouts = multiple_rollouts
        self.line_color = line_color
        self.line_alpha = line_alpha
        self.line_width = line_width
        self.weights = weights

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

            for policy_name, (fn, mask) in policy_masks.items():
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
            # Create a completely new figure and axis for each environment to prevent carryover
            plt.close(
                "all"
            )  # Close all existing figures first to prevent memory leaks

            # Initialize a new figure for each environment
            fig = plt.figure(figsize=self.figsize)

            # Create a new axis with proper projection
            if self.render_3d:
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=30, azim=45)  # Set default 3D view angle
            else:
                ax = fig.add_subplot(111)

            # Set up the figure and axis
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_aspect("equal", adjustable="box")

            # Add to figures list - use a copy to ensure it's detached from the current matplotlib state
            figs.append(fig)

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

            if plot_guidance_pos_xy:
                self._plot_reference_xy(
                    ax=ax,
                    control_mask=controlled_live,
                    env_idx=env_idx,
                    trajectory=self.trajectory,
                    line_width_scale=line_width_scale,
                    plot_guidance_up_to_time=plot_guidance_up_to_time,
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
                self.plot_agent_trajectories(
                    ax=ax,
                    agent_positions=agent_positions,
                    env_idx=env_idx,
                    time_step=time_step,
                    controlled_live=controlled_live,
                    render_3d=self.render_3d,
                    colorbar=colorbar,
                    marker_scale=marker_scale,
                    multiple_rollouts=self.multi_rollouts,
                    line_color=self.line_color,
                    line_alpha=self.line_alpha,
                    line_width=self.line_width,
                    weights=self.weights,
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

            time_step = (
                self.env_config.episode_len
                - self.sim_object.steps_remaining_tensor().to_torch()[
                    env_idx, 0
                ]
            ).item()

            # Add time step text to the figure
            ax.text(
                0.05,  # x position in axes coordinates (5% from left)
                0.95,  # y position in axes coordinates (95% from bottom)
                f"t = {time_step}",
                transform=ax.transAxes,  # Use axes coordinates
                fontsize=15,
                color="black",
                ha="left",
                va="top",
            )

            time_step = (
                self.env_config.episode_len
                - self.sim_object.steps_remaining_tensor().to_torch()[
                    env_idx, 0
                ]
            ).item()

            # Add time step text to the figure
            ax.text(
                0.05,  # x position in axes coordinates (5% from left)
                0.95,  # y position in axes coordinates (95% from bottom)
                f"t = {time_step}",
                transform=ax.transAxes,  # Use axes coordinates
                fontsize=15,
                color="black",
                ha="left",
                va="top",
            )

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

            # Apply tight layout to current figure
            fig.tight_layout(pad=2, rect=[0.00, 0.00, 0.9, 1])

            # Close the figure to prevent memory leaks and cleanup
            plt.close(fig)

        return figs

    def plot_agent_trajectories(
        self,
        ax,
        agent_positions,
        env_idx=0,
        time_step=None,
        controlled_live=None,
        render_3d=False,
        colorbar=True,
        marker_scale=1,
        line_color=None,
        line_alpha=None,
        line_width=None,
        multiple_rollouts=True,
        weights=None,
    ):
        """
        Plot agent trajectories on the given matplotlib axis.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to plot on
        agent_positions : torch.Tensor
            Tensor of shape [env, rollouts, agent, time, 2] or
            [rollouts, agent, time, 2] (legacy format)
        env_idx : int, default=0
            Index of the environment to plot
        time_step : int, optional
            Current time step to plot trajectories up to. If None, uses all timesteps.
        controlled_live : list or tensor, optional
            Boolean mask indicating which agents are controlled and alive.
            If None, assumes all agents are active.
        render_3d : bool, default=False
            Whether to render in 3D
        colorbar : bool, default=True
            Whether to add a colorbar
        marker_scale : float, default=1
            Scale factor for markers and text
        line_color : str, optional
            Color for trajectory lines (only used for multiple rollouts mode without weights)
        line_alpha : float, optional
            Transparency of trajectory lines
        line_width : float, optional
            Width of trajectory lines
        multiple_rollouts : bool, default=True
            Whether agent_positions contains multiple rollouts
        weights : list, optional
            Used to color lines by the absolute magnitude of weights.

        Returns:
        --------
        matplotlib.cm.ScalarMappable or None: Returns the color mapper
        """
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        from matplotlib.collections import LineCollection

        if render_3d:
            try:
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
            except ImportError:
                print(
                    "Warning: 3D rendering requested but mplot3d not available. Falling back to 2D."
                )
                render_3d = False

        if agent_positions is None:
            return None

        # Determine shape format and extract appropriate data
        if len(agent_positions.shape) == 5:  # [env, rollouts, agent, time, 2]
            # New format with environment dimension first
            if env_idx >= agent_positions.shape[0]:
                print(
                    f"Warning: env_idx {env_idx} out of range. Using env_idx=0"
                )
                env_idx = 0

            # Extract the specific environment data
            positions_to_plot = agent_positions[
                env_idx
            ]  # Shape: [rollouts, agent, time, 2]
            n_rollouts, n_agents, n_steps, _ = positions_to_plot.shape

        elif len(agent_positions.shape) == 4:  # [rollouts, agent, time, 2]
            # Legacy format without environment dimension
            positions_to_plot = agent_positions
            n_rollouts, n_agents, n_steps, _ = positions_to_plot.shape
            print("Warning: Using legacy format without environment dimension")
        else:
            raise ValueError(
                f"Unexpected shape for agent_positions: {agent_positions.shape}"
            )

        # Set defaults for multiple rollouts mode
        line_color = line_color or "#1f77b4"  # Default blue
        line_alpha = 0.3 if line_alpha is None else line_alpha
        line_width = 1 if line_width is None else line_width

        # If time_step is not provided, use all timesteps
        if time_step is None:
            time_step = n_steps

        # If controlled_live is not provided, assume all agents are active
        if controlled_live is None:
            controlled_live = [True] * n_agents

        # Setup for weight-based coloring
        if weights is not None:
            weight_values = np.array(
                weights
            )  # Use absolute values for coloring

            # Set up a colormap for weights
            weight_cmap = plt.cm.coolwarm
            weight_norm = plt.Normalize(
                vmin=weight_values.min().item(),
                vmax=weight_values.max().item(),
            )
            weight_sm = plt.cm.ScalarMappable(
                cmap=weight_cmap, norm=weight_norm
            )

        # Process each rollout and each agent
        for rollout_idx in range(n_rollouts):
            for agent_idx in range(n_agents):
                if not controlled_live[agent_idx]:
                    continue

                # Get trajectory for this rollout and agent
                trajectory = positions_to_plot[
                    rollout_idx, agent_idx, :time_step, :
                ]

                # Create valid mask
                valid_mask = (
                    (trajectory[:, 0] != 0)
                    & (trajectory[:, 1] != 0)
                    & (torch.abs(trajectory[:, 0]) < OUT_OF_BOUNDS)
                    & (torch.abs(trajectory[:, 1]) < OUT_OF_BOUNDS)
                )

                # Get valid trajectory points
                valid_trajectory = trajectory[valid_mask]

                # Only proceed if we have at least 2 points
                if len(valid_trajectory) > 1:
                    points = valid_trajectory.cpu().numpy()

                    # Determine color based on weights or fixed color
                    if weights is not None:
                        # Use the weight value for this rollout
                        weight_value = weight_values[rollout_idx].item()
                        segment_color = weight_cmap(weight_norm(weight_value))
                    else:
                        segment_color = line_color

                    if render_3d:
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

                        # Create line collection for 3D
                        lc = Line3DCollection(
                            segments_3d,
                            colors=segment_color,
                            linewidths=line_width,
                            alpha=line_alpha,
                            zorder=1,
                        )
                        ax.add_collection3d(lc)
                    else:
                        segments = []
                        for i in range(len(points) - 1):
                            segment = np.array(
                                [
                                    [points[i][0], points[i][1]],
                                    [points[i + 1][0], points[i + 1][1]],
                                ]
                            )
                            segments.append(segment)

                        # Create line collection for 2D
                        lc = LineCollection(
                            segments,
                            colors=segment_color,
                            linewidths=line_width,
                            alpha=line_alpha,
                            zorder=1,
                        )
                        ax.add_collection(lc)

        # Add colorbar for weight-based coloring - outside the loops
        if weights is not None and colorbar:
            try:
                fig = ax.get_figure()
                # Create horizontal colorbar at the bottom
                # Parameters: [left, bottom, width, height]
                cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
                cbar = fig.colorbar(
                    weight_sm, cax=cbar_ax, orientation="horizontal"
                )
                cbar.set_label(
                    f"Conditioning param value", fontsize=15 * marker_scale
                )
                cbar.ax.tick_params(labelsize=12 * marker_scale)
            except Exception as e:
                print(f"Warning: Could not add colorbar: {e}")

            return weight_sm

        return None

    def _plot_reference_xy(
        self,
        ax: matplotlib.axes.Axes,
        env_idx: int,
        control_mask: torch.Tensor,
        trajectory,
        line_width_scale: int = 1.0,
        plot_guidance_up_to_time: bool = False,
    ):
        """Plot the log replay trajectory for controlled agents in either 2D or 3D."""
        if self.render_3d:
            # Get trajectory points - make a clean copy to avoid reference issues
            try:
                trajectory_points = (
                    trajectory.pos_xy[env_idx, control_mask, :, :]
                    .clone()
                    .numpy()
                )

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

                        # Create a fresh line collection for each plot
                        lc = Line3DCollection(
                            segments,
                            colors=colors,
                            linewidth=2 * line_width_scale,
                        )
                        ax.add_collection3d(lc)

                        # Add points at trajectory positions - creating a new scatter object each time
                        ax.scatter3D(
                            valid_points[:, 0],
                            valid_points[:, 1],
                            np.full_like(
                                valid_points[:, 0], trajectory_height
                            ),
                            color="lightgreen",
                            s=10,
                            alpha=0.5,
                            zorder=0,
                        )
            except Exception as e:
                print(f"Error plotting 3D reference trajectory: {e}")
        else:
            try:

                if plot_guidance_up_to_time:
                    # Get the time step for the current environment
                    time_step = (
                        self.env_config.episode_len
                        - self.sim_object.steps_remaining_tensor().to_torch()[
                            env_idx, 0
                        ]
                    ).item()

                    # Limit the trajectory to the specified time step
                    # Create a new scatter plot for this specific environment and control mask
                    pos_x_context = (
                        trajectory.pos_xy.clone()[
                            env_idx, control_mask, :time_step, 0
                        ]
                        .cpu()
                        .numpy()
                    )
                    pos_y_context = (
                        trajectory.pos_xy.clone()[
                            env_idx, control_mask, :time_step, 1
                        ]
                        .cpu()
                        .numpy()
                    )

                    # Filter out invalid points (zeros or out of bounds)
                    valid_mask = (
                        (pos_x_context != 0)
                        & (pos_y_context != 0)
                        & (np.abs(pos_x_context) < OUT_OF_BOUNDS)
                        & (np.abs(pos_y_context) < OUT_OF_BOUNDS)
                    )

                    ax.scatter(
                        pos_x_context,
                        pos_y_context,
                        color="#f4a261",
                        linewidth=0.1 * line_width_scale,
                        alpha=0.7,
                        s=25,
                        zorder=0,
                    )

                    pos_x = (
                        trajectory.pos_xy.clone()[
                            env_idx, control_mask, time_step:, 0
                        ]
                        .cpu()
                        .numpy()
                    )
                    pos_y = (
                        trajectory.pos_xy.clone()[
                            env_idx, control_mask, time_step:, 1
                        ]
                        .cpu()
                        .numpy()
                    )

                    # Filter out invalid points (zeros or out of bounds)
                    valid_mask = (
                        (pos_x != 0)
                        & (pos_y != 0)
                        & (np.abs(pos_x) < OUT_OF_BOUNDS)
                        & (np.abs(pos_y) < OUT_OF_BOUNDS)
                    )

                    # Apply mask if any valid points exist
                    if np.any(valid_mask):
                        pos_x = pos_x[valid_mask]
                        pos_y = pos_y[valid_mask]

                    ax.scatter(
                        pos_x,
                        pos_y,
                        color="g",
                        s=25,
                        alpha=0.25,
                        zorder=0,
                    )

                else:
                    pos_x = (
                        trajectory.pos_xy.clone()[env_idx, control_mask, :, 0]
                        .cpu()
                        .numpy()
                    )
                    pos_y = (
                        trajectory.pos_xy.clone()[env_idx, control_mask, :, 1]
                        .cpu()
                        .numpy()
                    )

                    # Filter out invalid points (zeros or out of bounds)
                    valid_mask = (
                        (pos_x != 0)
                        & (pos_y != 0)
                        & (np.abs(pos_x) < OUT_OF_BOUNDS)
                        & (np.abs(pos_y) < OUT_OF_BOUNDS)
                    )

                    # Apply mask if any valid points exist
                    if np.any(valid_mask):
                        pos_x = pos_x[valid_mask]
                        pos_y = pos_y[valid_mask]

                    # Create a fresh scatter plot
                    ax.scatter(
                        pos_x,
                        pos_y,
                        color="g",
                        s=25,
                        alpha=0.1,
                        zorder=0,
                    )
            except Exception as e:
                print(f"Error plotting 2D reference trajectory: {e}")

    def _plot_vbd_trajectory(
        self,
        ax: matplotlib.axes.Axes,
        env_idx: int,
        control_mask: torch.Tensor,
        vbd_trajectory: VBDTrajectory,
        line_width_scale: int = 1.0,
    ):
        """Plot the VBD trajectory for controlled agents in either 2D or 3D."""
        if self.render_3d:
            # Get trajectory points
            trajectory_points = vbd_trajectory.pos_xy[
                env_idx, control_mask, :, :
            ].numpy()

            # Set a fixed height for trajectory visualization
            trajectory_height = 0.05

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
                vbd_trajectory.pos_xy[env_idx, control_mask, :, 0].numpy(),
                vbd_trajectory.pos_xy[env_idx, control_mask, :, 1].numpy(),
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
                            alpha=0.7,
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
                            alpha=0.6,
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
        plot_goal_points: bool = False,
        line_width_scale: int = 1.0,
        marker_size_scale: int = 1.0,
        extended_goals: Optional[Dict[str, torch.Tensor]] = None,
        world_based_policy_mask: Optional[
            Dict[int, Dict[str, torch.Tensor]]
        ] = None,
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

        def plot_agent_group_2d(bboxes, color, by_policy=False):
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
                (is_ok_mask, "#f4a261"),  # AGENT_COLOR_BY_STATE["ok"]),
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
                        linewidth=2.0 * line_width_scale,
                        c=color,
                        marker="o",
                        zorder=3,
                    )
                    for x, y in zip(goal_x, goal_y):
                        circle = Circle(
                            (x, y),
                            radius=self.goal_radius,
                            color=color,
                            fill=False,
                            linestyle="--",
                            linewidth=3 * line_width_scale,
                            zorder=3,
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

        if not world_based_policy_mask:  ## controlled by the same policy
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
            for policy_name, mask in policy_mask.items():

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
                bboxes_controlled_ok, AGENT_COLOR_BY_POLICY, by_policy=True
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

    def plot_agent_observation(
        self,
        agent_idx: int,
        env_idx: int,
        figsize: Tuple[int, int] = (10, 10),
        trajectory: Optional[np.ndarray] = None,
        step_reward: Optional[float] = None,
        route_progress: Optional[float] = None,
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

        observation_ego = None
        if self.env_config.ego_state:
            observation_ego = LocalEgoState.from_tensor(
                self_obs_tensor=self.sim_object.self_observation_tensor(),
                backend=self.backend,
                device="cpu",
            )

        observation_roadgraph = None
        if self.env_config.road_map_obs:
            observation_roadgraph = LocalRoadGraphPoints.from_tensor(
                local_roadgraph_tensor=self.sim_object.agent_roadmap_tensor(),
                backend=self.backend,
                device="cpu",
            )

        observation_partner = None
        if self.env_config.partner_obs:
            observation_partner = PartnerObs.from_tensor(
                partner_obs_tensor=self.sim_object.partner_observations_tensor(),
                backend=self.backend,
                device="cpu",
            )

        lidar_obs = (
            None  # Note: Lidar obs are in global coordinates by default
        )
        if self.env_config.lidar_obs:
            lidar_obs = LidarObs.from_tensor(
                lidar_tensor=self.sim_object.lidar_tensor(),
                backend=self.backend,
                device="cpu",
            )

        marker_scale = max(figsize) / 15
        line_width_scale = max(figsize) / 15

        fig, ax = plt.subplots(figsize=figsize)
        self._cleanup_axis(ax)

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
                    s=8 * marker_scale,
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
                        linewidth=line_width_scale,
                    )
                    ax.plot(
                        [x_start + width_dx, x_end + width_dx],
                        [y_start + width_dy, y_end + width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=line_width_scale,
                    )
                    ax.plot(
                        [x_start - width_dx, x_start + width_dx],
                        [y_start - width_dy, y_start + width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=line_width_scale,
                    )
                    ax.plot(
                        [x_end - width_dx, x_end + width_dx],
                        [y_end - width_dy, y_end + width_dy],
                        color=ROAD_GRAPH_COLORS[road_type],
                        alpha=0.5,
                        linewidth=line_width_scale,
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
                line_width_scale=line_width_scale * 2.0,
            )

        if observation_ego is not None:
            # Check if agent index is valid, otherwise return None
            if observation_ego.id[env_idx, agent_idx] == -1:
                return None, None

            ego_agent_color = (
                "r"
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
                line_width_scale=2.3,
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
                fc="k",
                ec="k",
                zorder=10,
            )

        if lidar_obs is not None:
            num_lidar_samples = lidar_obs.num_lidar_samples * 3

            ego_lidar_pos_xy = (
                lidar_obs.all_lidar_samples[env_idx, agent_idx, :, :, 2:4]
                .flatten(end_dim=1)
                .cpu()
                .numpy()
            )

            ego_lidar_entity_types = (
                lidar_obs.all_lidar_samples[env_idx, agent_idx, :, :, 1]
                .flatten()
                .cpu()
                .numpy()
            )

            for lidar_sample_idx in range(num_lidar_samples):
                ax.scatter(
                    ego_lidar_pos_xy[lidar_sample_idx, 0],
                    ego_lidar_pos_xy[lidar_sample_idx, 1],
                    s=2,
                    marker="o",
                    c="k",
                    alpha=0.5,
                )

        time_step = (
            self.env_config.episode_len
            - self.sim_object.steps_remaining_tensor().to_torch()[
                env_idx, agent_idx
            ]
        ).item()

        ax.text(
            0.05,
            0.90,
            r"$O_{t}$ for " + f"t = {time_step}",
            transform=ax.transAxes,
            fontsize=15,
            color="black",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=1.0, edgecolor="none", pad=3),
        )

        if step_reward is not None:
            reward_color = (
                "g" if step_reward > 0 else "r" if step_reward < 0 else "black"
            )

            ax.text(
                0.05,
                0.85,
                r"$R_{t+1} = $" + f"{step_reward:.3f}",
                transform=ax.transAxes,
                fontsize=15,
                color=reward_color,
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="white", alpha=1.0, edgecolor="none", pad=3
                ),
            )

        if route_progress is not None:
            ax.text(
                0.05,
                0.80,
                f"Route progress = {route_progress:.2f}",
                transform=ax.transAxes,
                fontsize=15,
                color="black",
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="white", alpha=1.0, edgecolor="none", pad=3
                ),
            )

        if trajectory is not None and len(trajectory) > 0:
            mask = trajectory[:, 0] != constants.INVALID_ID
            # Plot the trajectory as a line
            ax.scatter(
                trajectory[:, 0][mask].cpu(),  # x coordinates
                trajectory[:, 1][mask].cpu(),  # y coordinates
                color="g",
                linewidth=0.01 * line_width_scale,
                marker="o",
                alpha=0.6,
                zorder=0,
            )

            # Draw a circle around every point in the trajectory
            for i in range(trajectory.shape[0]):
                if mask[i]:
                    # Draw a circle around the trajectory point
                    circle = Circle(
                        (trajectory[i, 0].cpu(), trajectory[i, 1].cpu()),
                        radius=self.env_config.guidance_pos_xy_radius,
                        color="#d4a373",
                        fill=False,
                        linestyle="--",
                        alpha=0.3,
                    )
                    ax.add_patch(circle)

        ax.set_xlim((-self.env_config.obs_radius, self.env_config.obs_radius))
        ax.set_ylim((-self.env_config.obs_radius, self.env_config.obs_radius))

        # Add a circle representing the observation radius
        observation_circle = Circle(
            (0, 0),  # Center at origin
            radius=self.env_config.obs_radius,
            color="black",
            fill=False,
            linestyle="-",
            linewidth=1.0,
            alpha=0.7,
        )

        ax.add_patch(observation_circle)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis("off")

        return fig

    def _cleanup_axis(self, ax):
        """Clean up all collections and artists from the axis."""
        if self.render_3d:
            # Clean 3D collections
            for collection in ax.collections[:]:
                collection.remove()

            # Clean lines
            for line in ax.lines[:]:
                line.remove()
        else:
            # Clean 2D collections
            for collection in ax.collections[:]:
                collection.remove()

            # Clean patches
            for patch in ax.patches[:]:
                patch.remove()

            # Clean lines
            for line in ax.lines[:]:
                line.remove()

            # Clean texts
            for text in ax.texts[:]:
                text.remove()
