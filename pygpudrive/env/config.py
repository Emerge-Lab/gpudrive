"""Configs for GPU Drive Environments."""

from dataclasses import dataclass
import torch
from enum import Enum
from typing import Tuple

class RenderMode(Enum):
    PYGAME_ABSOLUTE = "pygame_absolute"
    PYGAME_EGOCENTRIC = "pygame_egocentric"
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
    render_mode: RenderMode
    view_option: Enum
    resolution: Tuple[int, int]

    def __str__(self):
        return f"RenderMode: {self.render_mode.value}, ViewOption: {self.view_option.value}, Resolution: {self.resolution}"

@dataclass
class EnvConfig:
    """Configurations for gpudrive gym environment."""

    # Observation space
    ego_state: bool = True  # Ego vehicle state
    road_map_obs: bool = False  # Road graph
    partner_obs: bool = False  # Partner vehicle info

    # Normalize
    normalize_obs: bool = False

    # Action space
    steer_actions: torch.Tensor = torch.tensor([-0.6, 0, 0.6])
    accel_actions: torch.Tensor = torch.tensor([-3, 0, 3])

    # Collision behavior
    collision_behavior: str = "remove"  # options: "remove", "stop", "ignore"

    # Reward
    dist_to_goal_threshold: float = 3.0

    """Constants to normalize observations."""
    # Values to normalize by: Ego state
    max_speed: int = 100
    max_veh_len: int = 25
    max_veh_width: int = 5
    max_rel_goal_coord: int = 100

    # TODO: Values to normalize by: Partner state
    max_partner: int = 50
