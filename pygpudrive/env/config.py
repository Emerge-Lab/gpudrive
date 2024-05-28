"""Configs for GPU Drive Environments."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class EnvConfig:
    """Configurations for gpudrive gym environment."""

    # Observation space
    ego_state: bool = True  # Ego vehicle state
    road_map_obs: bool = False  # Road graph
    partner_obs: bool = False  # Partner vehicle info

    # Road observation algorithm
    road_obs_algorithm: str = "k_nearest_roadpoints"
    obs_radius: float = 10.0

    # Action space (discrete)
    steer_actions: torch.Tensor = torch.tensor(
        [-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6]
    )
    accel_actions: torch.Tensor = torch.tensor([-3, -1, 0, 1, 3])

    # Collision behavior
    collision_behavior: str = "remove"  # options: "remove", "stop", "ignore"
    # Remove all non vehicles (bicylces, pedestrians) from the scene
    remove_non_vehicles: bool = True

    # Reward
    reward_type: str = (
        "sparse_on_goal_achieved"  # options: "sparse_on_goal_achieved"
    )
    dist_to_goal_threshold: float = 3.0

    """Constants to normalize observations."""
    norm_obs: bool = False

    # Values to normalize by: Ego state
    max_speed: int = 100
    max_veh_len: int = 25
    max_veh_width: int = 5
    min_rel_goal_coord: int = -100
    max_rel_goal_coord: int = 100
    min_rel_agent_pos: int = -100
    max_rel_agent_pos: int = 100
    max_orientation_rad: float = 2 * np.pi
    min_rm_coord: int = -300
    max_rm_coord: int = 300

    # Datasete settings
    # first_n - Takes the first num_worlds files. Fails if num files < num_worlds.
    # random_n - Takes num_worlds files randomly. Fails if num files < num_worlds.
    # pad_n - Initializes as many files as possible first.
    # Then it repeats the first file to pad until num_worlds
    # files are loaded. Will fail if the number of files are more than num_worlds.
    # exact_n - Init exactly num_worlds files.
    sample_method: str = "first_n"

    # Related to settings
    eval_expert_mode: bool = (
        False  # Set this to true if you want to return all agent info
    )
