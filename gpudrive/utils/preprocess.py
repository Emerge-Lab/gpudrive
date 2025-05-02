import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Union
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from gpudrive.datatypes.trajectory import (
    LogTrajectory,
    VBDTrajectory,
    VBDTrajectoryOnline,
)


def smooth_scenario(
    reference_trajectory: Union[
        LogTrajectory, VBDTrajectory, VBDTrajectoryOnline
    ],
    window_size: int = 11,
    polyorder: int = 3,
):
    """
    Smooths the trajectory of a scenario by applying filtering and interpolation techniques.

    Args:
        reference_trajectory: The trajectory to be smoothed. It can be of type LogTrajectory, VBDTrajectory, or VBDTrajectoryOnline.
        The trajectory should contain the following attributes:
            - vel_xy: A tensor of shape (num_worlds, num_agents, traj_len, 2) representing the velocity in x and y directions.
            - yaw: A tensor of shape (num_worlds, num_agents, traj_len, 1) representing the yaw angle.
            - valids: A tensor of shape (num_worlds, num_agents, traj_len, 1) representing the validity of each timestep.
        window_size (int, optional): The size of the window used for smoothing. Defaults to 11.
        polyorder (int, optional): The order of the polynomial used for smoothing. Defaults to 3.

    Returns:
        the updated reference_trajectory object.
    """
    original_valid = reference_trajectory.valids.bool().cpu().numpy()

    num_worlds, num_agents, num_timesteps = (
        original_valid.shape[0],
        original_valid.shape[1],
        original_valid.shape[2],
    )

    # Elements that we want to smooth
    vel = np.concatenate(
        [
            reference_trajectory.vel_xy.cpu().numpy(),
            np.sin(reference_trajectory.yaw.cpu().numpy()),
            np.cos(reference_trajectory.yaw.cpu().numpy()),
        ],
        axis=-1,
    )

    smoothed_vel_xy = np.zeros_like(reference_trajectory.vel_xy)
    smoothed_yaw = np.zeros_like(reference_trajectory.yaw)
    smoothed_valid = np.zeros_like(original_valid, dtype=bool)

    t = np.arange(num_timesteps)

    for world_idx in tqdm(
        range(num_worlds), colour="blue", desc="Smoothing guidance data"
    ):
        for agent_idx in range(num_agents):

            # Extract raw data and valid mask
            valid = original_valid[world_idx, agent_idx].squeeze(-1)

            t_valid = t[valid]
            vel_valid = vel[world_idx, agent_idx][valid]
            valid_idx = np.where(valid)[0]

            if valid.sum() <= window_size:
                continue

            # Use zscore to filter out outliers
            std = np.clip(
                np.std(vel_valid, axis=-2, keepdims=True),
                a_min=0.1,
                a_max=None,
            )
            mean = np.mean(vel_valid, axis=-2, keepdims=True)
            z = np.abs((vel_valid - mean) / std)
            filtered_idx = np.all(z < 4, axis=-1)
            valid_idx = valid_idx[filtered_idx]

            if len(valid_idx) == 0:  # skip if no valid data
                continue

            first_valid_idx = valid_idx[0]
            last_valid_idx = valid_idx[-1]
            if (last_valid_idx - first_valid_idx) <= 3:
                continue

            # Extract valid velocity data and interpolate
            t_valid = t[valid_idx]

            vel_valid = vel[world_idx, agent_idx][valid_idx, :]
            vel_interp = interp1d(t_valid, vel_valid, axis=0, kind="linear")

            t_interped = np.arange(first_valid_idx, last_valid_idx + 1)
            vel_interped = vel_interp(t_interped)

            # Smooth the interpolated data
            vel_smoothed = savgol_filter(
                vel_interped,
                min(last_valid_idx - first_valid_idx, window_size),
                polyorder,
                axis=0,
            )

            # Store
            smoothed_vel_xy[world_idx, agent_idx][
                first_valid_idx : last_valid_idx + 1
            ] = vel_smoothed[:, :2]
            smoothed_yaw[world_idx, agent_idx][
                first_valid_idx : last_valid_idx + 1
            ] = np.expand_dims(
                np.arctan2(vel_smoothed[:, 2], vel_smoothed[:, 3]), axis=-1
            )
            smoothed_valid[world_idx, agent_idx][
                first_valid_idx : last_valid_idx + 1
            ] = True

    # Update the reference trajectory with smoothed velocity and yaw
    reference_trajectory.vel_xy = torch.from_numpy(smoothed_vel_xy).to(
        reference_trajectory.vel_xy.device
    )
    reference_trajectory.yaw = torch.from_numpy(smoothed_yaw).to(
        reference_trajectory.yaw.device
    )
    reference_trajectory.valids = torch.from_numpy(smoothed_valid).to(
        reference_trajectory.valids.device
    )

    return reference_trajectory
