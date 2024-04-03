"""Configs for Gpudrive Environments."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configurations for Gpudrive Environment."""

    # Observation space
    ego_state: bool = True
    vis_obs: bool = False
    partner_obs: bool = False
    road_map_obs: bool = False
