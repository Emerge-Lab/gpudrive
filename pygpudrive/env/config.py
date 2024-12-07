# pylint: skip-file
"""Configuration classes and enums for GPUDrive Environments."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import torch

import gpudrive


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
    norm_obs: bool = True  # Normalize observations

    # NOTE: If disable_classic_obs is True, ego_state, road_map_obs,
    # and partner_obs are invalid. This makes the sim 2x faster
    disable_classic_obs: bool = False  # Disable classic observations
    lidar_obs: bool = False  # Use LiDAR in observations

    # Set the weights for the reward components
    # R = a * collided + b * goal_achieved + c * off_road
    collision_weight: float = 0.0
    goal_achieved_weight: float = 1.0
    off_road_weight: float = 0.0

    # Road observation algorithm settings
    road_obs_algorithm: str = "linear"  # Algorithm for road observations
    obs_radius: float = 100.0  # Radius for road observations
    polyline_reduction_threshold: float = (
        1.0  # Threshold for polyline reduction
    )

    # Dynamics model
    dynamics_model: str = (
        "classic"  # Options: "classic", "bicycle", "delta_local", or "state"
    )

    # Action space settings (if discretized)
    # Classic or Invertible Bicycle dynamics model
    steer_actions: torch.Tensor = torch.round(
        torch.linspace(-torch.pi, torch.pi, 36), decimals=3
    )
    accel_actions: torch.Tensor = torch.round(
        torch.linspace(-4.0, 4.0, 16), decimals=3
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
    collision_behavior: str = "remove"  # Options: "remove", "stop", "ignore"

    # Scene configuration
    remove_non_vehicles: bool = True  # Remove non-vehicle entities from scene

    # Reward settings
    reward_type: str = (
        "sparse_on_goal_achieved"  # Alternatively, "weighted_combination"
    )

    dist_to_goal_threshold: float = (
        3.0  # Radius around goal considered as "goal achieved"
    )

    # C++ and Python shared settings (modifiable via C++ codebase)
    max_num_agents_in_scene: int = (
        gpudrive.kMaxAgentCount
    )  # Max number of objects in simulation
    max_num_rg_points: int = (
        gpudrive.kMaxRoadEntityCount
    )  # Max number of road graph segments
    roadgraph_top_k: int = (
        gpudrive.kMaxAgentMapObservationsCount
    )  # Top-K road graph segments agents can view
    episode_len: int = (
        gpudrive.episodeLen
    )  # Length of an episode in the simulator
    num_lidar_samples: int = gpudrive.numLidarSamples


    #Param to init all objects:
    init_all_objects: bool = False

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

    path: str
    num_scenes: int
    discipline: SelectionDiscipline = SelectionDiscipline.PAD_N
    k_unique_scenes: Optional[int] = None
    seed: Optional[int] = None


class RenderMode(Enum):
    """Enum for specifying rendering mode."""

    PYGAME_ABSOLUTE = "pygame_absolute"
    PYGAME_EGOCENTRIC = "pygame_egocentric"
    PYGAME_LIDAR = "pygame_lidar"
    MADRONA_RGB = "madrona_rgb"
    MADRONA_DEPTH = "madrona_depth"


class PygameOption(Enum):
    """Enum for Pygame rendering options."""

    HUMAN = "human"
    RGB = "rgb"


class MadronaOption(Enum):
    """Enum for Madrona rendering options."""

    AGENT_VIEW = "agent_view"
    TOP_DOWN = "top_down"


@dataclass
class RenderConfig:
    """Configuration settings for rendering the environment.

    Attributes:
        render_mode (RenderMode): The mode used for rendering the environment.
        view_option (Enum): Rendering view option (e.g., RGB, human view).
        resolution (Tuple[int, int]): Resolution of the rendered image.
        line_thickness (int): Thickness of the road lines in the rendering.
        draw_obj_idx (bool): Whether to draw object indices on objects.
        obj_idx_font_size (int): Font size for object indices.
        color_scheme (str): Color mode for the rendering ("light" or "dark").
    """

    render_mode: RenderMode = RenderMode.PYGAME_ABSOLUTE
    view_option: Enum = PygameOption.RGB
    resolution: Tuple[int, int] = (1024, 1024)
    line_thickness: int = 0.7
    draw_obj_idx: bool = False
    obj_idx_font_size: int = 9
    color_scheme: str = "light"

    def __str__(self) -> str:
        """Returns a string representation of the rendering configuration."""
        return (
            f"RenderMode: {self.render_mode.value}, ViewOption: {self.view_option.value}, "
            f"Resolution: {self.resolution}, LineThickness: {self.line_thickness}, "
            f"DrawObjectIdx: {self.draw_obj_idx}, ObjectIdxFontSize: {self.obj_idx_font_size}, "
            f"ColorScheme: {self.color_scheme}"
        )
