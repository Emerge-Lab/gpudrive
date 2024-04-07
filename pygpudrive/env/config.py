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
    goal_dist: bool = True
    abs_agent_pos: bool = False

    # Normalize
    normalize_obs: bool = False
    max_norm_by: int = 90  # Hardcoded normalization value
    min_norm_by: int = -90
    max_abs_coord: int = 9100  # Hardcoded normalization value
    min_abs_coord: int = -9100

    # Action space
    steer_actions: torch.Tensor = torch.tensor([-0.6, 0, 0.6])
    accel_actions: torch.Tensor = torch.tensor([-3, 0, 3])

    # Reward
    # TODO
