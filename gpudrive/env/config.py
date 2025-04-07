# pylint: skip-file
"""Configuration classes and enums for GPUDrive Environments."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import torch

import madrona_gpudrive


@dataclass
class EnvConfig:
    """Configuration settings for the GPUDrive gym environment.

    This class contains both Python-specific configurations and settings that
    are shared between Python and C++ components of the simulator.

    To modify simulator settings shared with C++, follow these steps:
    1. Navigate to `src/consts.hpp` in the C++ codebase.
    2. Locate and modify the constant (e.g., `kMaxAgentCount`).
    3. Save the changes to `src/consts.hpp`.
    4. Recompile the code to apply changes across both C++ and Python.
    """

    # Python-specific configurations
    # Observation space settings
    ego_state: bool = True  # Include ego vehicle state in observations
    road_map_obs: bool = True  # Include road graph in observations
    partner_obs: bool = True  # Include partner vehicle info in observations
    bev_obs: bool = False  # Include rasterized Bird's Eye View observations centered on ego vehicle
    norm_obs: bool = True  # Normalize observations

    # Maximum number of controlled agents in the scene
    max_controlled_agents: int = madrona_gpudrive.kMaxAgentCount
    num_worlds: int = 1  # Number of worlds in the environment

    # NOTE: If disable_classic_obs is True, ego_state, road_map_obs,
    # and partner_obs are invalid. This makes the sim 2x faster
    disable_classic_obs: bool = False  # Disable classic observations
    lidar_obs: bool = False  # Use LiDAR in observations

    # Road observation algorithm settings
    road_obs_algorithm: str = "linear"  # Algorithm for road observations
    obs_radius: float = 50.0  # Radius for road observations
    polyline_reduction_threshold: float = (
        0.1  # Threshold for polyline reduction
    )

    # Dynamics model
    dynamics_model: str = (
        "classic"  # Options: "classic", "bicycle", "delta_local", or "state"
    )

    # Action space settings (if discretized)
    # Classic or Invertible Bicycle dynamics model
    steer_actions: torch.Tensor = torch.round(
        torch.linspace(-torch.pi, torch.pi, 13), decimals=3
    )
    accel_actions: torch.Tensor = torch.round(
        torch.linspace(-4.0, 4.0, 7), decimals=3
    )
    head_tilt_actions: torch.Tensor = torch.Tensor([0])

    # Delta Local dynamics model
    dx: torch.Tensor = torch.round(torch.linspace(-2.0, 2.0, 20), decimals=3)
    dy: torch.Tensor = torch.round(torch.linspace(-2.0, 2.0, 20), decimals=3)
    dyaw: torch.Tensor = torch.round(
        torch.linspace(-3.14, 3.14, 20), decimals=3
    )

    # Global action space settings if StateDynamicsModel is used
    x: torch.Tensor = torch.round(
        torch.linspace(-100.0, 100.0, 10), decimals=3
    )
    y: torch.Tensor = torch.round(
        torch.linspace(-100.0, 100.0, 10), decimals=3
    )
    yaw: torch.Tensor = torch.round(
        torch.linspace(-3.14, 3.14, 10), decimals=3
    )
    vx: torch.Tensor = torch.round(torch.linspace(-10.0, 10.0, 10), decimals=3)
    vy: torch.Tensor = torch.round(torch.linspace(-10.0, 10.0, 10), decimals=3)

    # Collision behavior settings
    collision_behavior: str = "ignore"  # Options: "remove", "stop", "ignore"

    # Scene configuration
    remove_non_vehicles: bool = True  # Remove non-vehicle entities from scene

    # Initialization steps: Number of steps to take before the episode starts
    init_steps: int = 0

    # Goal behavior settings
    goal_behavior: str = "remove"  # Options: "stop", "ignore", "remove"
    add_goal_state: bool = True  # Add goal state to the scene

    # Reward settings
    reward_type: str = "sparse_on_goal_achieved"
    # Alternatively, "weighted_combination", "distance_to_logs", "distance_to_vdb_trajs", "reward_conditioned"

    # Weights for the reward components
    collision_weight: float = 0.0
    goal_achieved_weight: float = 1.0
    off_road_weight: float = 0.0

    condition_mode: str = "random"  # Options: "random", "fixed", "preset"

    # Define upper and lower bounds for reward components if using reward_conditioned
    collision_weight_lb: float = -1.0
    collision_weight_ub: float = 0.0
    goal_achieved_weight_lb: float = 1.0
    goal_achieved_weight_ub: float = 2.0
    off_road_weight_lb: float = -1.0
    off_road_weight_ub: float = 0.0

    dist_to_goal_threshold: float = (
        2.0  # Radius around goal considered as "goal achieved"
    )

    # C++ and Python shared settings (modifiable via C++ codebase)
    max_num_agents_in_scene: int = (
        madrona_gpudrive.kMaxAgentCount
    )  # Max number of objects in simulation
    max_num_rg_points: int = (
        madrona_gpudrive.kMaxRoadEntityCount
    )  # Max number of road graph segments
    roadgraph_top_k: int = (
        madrona_gpudrive.kMaxAgentMapObservationsCount
    )  # Top-K road graph segments agents can view
    episode_len: int = (
        madrona_gpudrive.episodeLen
    )  # Length of an episode in the simulator
    num_lidar_samples: int = madrona_gpudrive.numLidarSamples
    # Agent size scale factor (0.0-1.0)
    # Controls the visual and collision size of vehicles in the simulation.
    agent_size_scale: float = madrona_gpudrive.vehicleScale

    # Initialization mode
    init_mode: str = (
        "all_non_trivial"  # Options: all_non_trivial, all_objects, all_valid
    )

    # VBD model settings
    use_vbd: bool = False
    vbd_model_path: str = None
    vbd_trajectory_weight: float = 0.01
    vbd_in_obs: bool = False


class SelectionDiscipline(Enum):
    """Enum for selecting scenes discipline in dataset configuration."""

    FIRST_N = 0
    RANDOM_N = 1
    PAD_N = 2
    EXACT_N = 3
    K_UNIQUE_N = 4


@dataclass
class SceneConfig:
    """Configuration for selecting scenes from a dataset.

    Attributes:
        path (str): Path to the dataset.
        num_scenes (int): Number of scenes to select.
        discipline (SelectionDiscipline): Method for selecting scenes.
        k_unique_scenes (Optional[int]): Number of unique scenes if using
            K_UNIQUE_N discipline.
        seed (Optional[int]): Seed for random scene selection.
    """

    batch_size: int  # Number of scenes per batch (should be equal to number of worlds in the env).
    dataset_size: int  # Maximum number of files to include in the dataset.
    path: str = None
    num_scenes: int = None
    discipline: SelectionDiscipline = SelectionDiscipline.PAD_N
    k_unique_scenes: Optional[int] = None
    seed: Optional[int] = None


class RenderMode(Enum):
    """Enum for specifying rendering mode."""

    MATPLOTLIB = "matplotlib"
    MADRONA_RGB = "madrona_rgb"
    MADRONA_DEPTH = "madrona_depth"


class MadronaOption(Enum):
    """Enum for Madrona rendering options."""

    AGENT_VIEW = "agent_view"
    TOP_DOWN = "top_down"


@dataclass
class RenderConfig:
    """
    Configuration settings for rendering the environment.
    Attributes:
        render_mode (RenderMode): The mode used for rendering the environment.
        view_option (Enum): Rendering view option (e.g., RGB, human view).
        resolution (Tuple[int, int]): Resolution of the rendered image.
        draw_expert_trajectories (bool): Whether to draw expert trajectories.
        draw_only_controllable_veh (bool): Whether to draw only the trajectories of controllable vehicles.
        obj_idx_font_size (int): Font size for object indices.
        render_3d (bool): Whether to render the scene in 3D.
        vehicle_height (float): Height of the vehicle for 3D rendering.
    """

    render_mode: RenderMode = RenderMode.MATPLOTLIB
    view_option: Enum = None
    resolution: Tuple[int, int] = (1024, 1024)
    draw_expert_trajectories: bool = False
    draw_only_controllable_veh: bool = False
    obj_idx_font_size: int = 9
    render_3d: bool = False
    vehicle_height: float = 0.06
