"""Configs for GPUDrive Environments."""

from dataclasses import dataclass
import numpy as np
import torch
from enum import Enum
from typing import Tuple


@dataclass
class EnvConfig:
    """Configurations for gpudrive gym environment."""

    # Observation space
    ego_state: bool = True  # Ego vehicle state
    road_map_obs: bool = True  # Road graph
    partner_obs: bool = True  # Partner vehicle info
    norm_obs: bool = True
    enable_lidar: bool = False

    # Road observation algorithm
    road_obs_algorithm: str = "linear"
    obs_radius: float = 100.0
    polyline_reduction_threshold: float = 1.0

    # Action space (joint discrete)
    steer_actions: torch.Tensor = torch.round(
        torch.linspace(-1.0, 1.0, 13), decimals=3
    )
    accel_actions: torch.Tensor = torch.round(
        torch.linspace(-4.0, 4.0, 7), decimals=3
    )

    # Collision behavior
    collision_behavior: str = "remove"  # options: "remove", "stop", "ignore"

    # Remove all non vehicles (bicylces, pedestrians) from the scene
    remove_non_vehicles: bool = True

    # Reward
    reward_type: str = (
        "sparse_on_goal_achieved"  # options: "sparse_on_goal_achieved"
    )
    # The radius around the goal point within which the agent is considered
    # to have reached the goal
    dist_to_goal_threshold: float = 3.0

    # Maximum number of controlled vehicles and feature dimensions for network
    MAX_CONTROLLED_VEHICLES: int = 128
    ROADMAP_AGENT_FEAT_DIM: int = MAX_CONTROLLED_VEHICLES - 1
    TOP_K_ROADPOINTS: int = (
        200  # Number of visible roadpoints from the road graph
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # VALUES BELOW ARE ENV CONSTANTS: DO NOT CHANGE # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    max_speed: int = 100
    max_veh_len: int = 30
    max_veh_width: int = 10
    min_rel_goal_coord: int = -1000
    max_rel_goal_coord: int = 1000
    min_rel_agent_pos: int = -1000
    max_rel_agent_pos: int = 1000
    max_orientation_rad: float = 2 * np.pi
    min_rm_coord: int = -1000
    max_rm_coord: int = 1000
    max_road_line_segmment_len: int = 100
    max_road_scale: int = 100

    # DON'T CHANGE: Used for network
    EGO_STATE_DIM = 6 if ego_state else 0
    PARTNER_DIM = 10 if partner_obs else 0
    ROAD_MAP_DIM = 13 if road_map_obs else 0


class SelectionDiscipline(Enum):
    FIRST_N = 0
    RANDOM_N = 1
    PAD_N = 2
    EXACT_N = 3
    K_UNIQUE_N = 4


@dataclass
class SceneConfig:
    """Configurations for selecting scenes from a dataset."""

    path: str
    num_scenes: int
    discipline: SelectionDiscipline = SelectionDiscipline.PAD_N
    k_unique_scenes: int = None


class RenderMode(Enum):
    PYGAME_ABSOLUTE = "pygame_absolute"
    PYGAME_EGOCENTRIC = "pygame_egocentric"
    PYGAME_LIDAR = "pygame_lidar"
    MADRONA_RGB = "madrona_rgb"
    MADRONA_DEPTH = "madrona_depth"


class PygameOption(Enum):
    HUMAN = "human"
    RGB = "rgb"


class MadronaOption(Enum):
    AGENT_VIEW = "agent_view"
    TOP_DOWN = "top_down"


@dataclass
class RenderConfig:
    render_mode: RenderMode = RenderMode.PYGAME_ABSOLUTE
    view_option: Enum = PygameOption.RGB
    resolution: Tuple[int, int] = (1024, 1024)  # Quality of the rendered image
    line_thickness: int = 0.7  # Thickness of the road lines
    draw_obj_idx: bool = True  # Draw object index on the object

    def __str__(self):
        return f"RenderMode: {self.render_mode.value}, ViewOption: {self.view_option.value}, Resolution: {self.resolution}"


class SelectionDiscipline(Enum):
    FIRST_N = 0
    RANDOM_N = 1
    PAD_N = 2
    EXACT_N = 3
    K_UNIQUE_N = 4


@dataclass
class SceneConfig:
    path: str
    num_scenes: int
    discipline: SelectionDiscipline = SelectionDiscipline.PAD_N
    k_unique_scenes: int = None
