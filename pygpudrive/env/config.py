"""Configs for GPU Drive Environments."""

from dataclasses import dataclass
import torch


@dataclass
class EnvConfig:
    """Configurations for GPU Drive Environment."""

    # Observation space
    ego_state: bool = True  # Ego vehicle state
    road_map_obs: bool = False  # Road graph
    partner_obs: bool = False  # Partner vehicle info

    # Normalize
    normalize_obs: bool = False

    # Values to normalize by: Ego state
    max_speed: int = 100
    max_veh_len: int = 25
    max_veh_width: int = 5
    max_rel_goal_coord: int = 100

    # Action space
    steer_actions: torch.Tensor = torch.tensor([-0.6, 0, 0.6])
    accel_actions: torch.Tensor = torch.tensor([-3, 0, 3])

    # Collision behavior
    collision_behavior: str = "remove"  # options: "remove", "stop", "ignore"

    # Reward
    dist_to_goal_threshold: float = 3.0
