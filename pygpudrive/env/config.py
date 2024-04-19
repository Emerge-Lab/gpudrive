"""Configs for Gpudrive Environments."""

from dataclasses import dataclass
import torch


@dataclass
class EnvConfig:
    """Configurations for Gpudrive Environment."""

    # Observation space
    ego_state: bool = True
    vis_obs: bool = False
    partner_obs: bool = False
    road_map_obs: bool = False
    collision_state: bool = False
    goal_dist: bool = True

    # Normalize
    normalize_obs: bool = False
    max_speed: int = 20  # Speed is a positive value < 20
    max_veh_len: int = 25
    max_veh_width: int = 5
    max_rel_goal_coords: int = 200
    max_dist_to_goal: int = 200  # L2 norm to goal

    # Action space
    steer_actions: torch.Tensor = torch.tensor([-0.6, 0, 0.6])
    accel_actions: torch.Tensor = torch.tensor([-3, 0, 3])

    ## Collision behavior
    collision_behavior: str = "remove"  # options: "remove", "stop", "ignore"

    # Reward
    dist_to_goal_threshold: float = 3.0
