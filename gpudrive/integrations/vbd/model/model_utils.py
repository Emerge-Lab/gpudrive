import torch
import numpy as np


def batch_transform_trajs_to_local_frame(trajs, ref_idx=-1):
    """
    Batch transform trajectories to the local frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, T, x].
        ref_idx (int): Reference index for the local frame. Default is -1.

    Returns:
        torch.Tensor: Transformed trajectories in the local frame.

    """
    x = trajs[..., 0]
    y = trajs[..., 1]
    theta = trajs[..., 2]
    v_x = trajs[..., 3]
    v_y = trajs[..., 4]

    local_x = (x - x[:, :, ref_idx, None]) * torch.cos(
        theta[:, :, ref_idx, None]
    ) + (y - y[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None])
    local_y = -(x - x[:, :, ref_idx, None]) * torch.sin(
        theta[:, :, ref_idx, None]
    ) + (y - y[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None])

    local_theta = theta - theta[:, :, ref_idx, None]
    local_theta = wrap_angle(local_theta)

    local_v_x = v_x * torch.cos(theta[:, :, ref_idx, None]) + v_y * torch.sin(
        theta[:, :, ref_idx, None]
    )
    local_v_y = -v_x * torch.sin(theta[:, :, ref_idx, None]) + v_y * torch.cos(
        theta[:, :, ref_idx, None]
    )

    local_trajs = torch.stack(
        [local_x, local_y, local_theta, local_v_x, local_v_y], dim=-1
    )
    local_trajs[trajs[..., :5] == 0] = 0

    if trajs.shape[-1] > 5:
        trajs = torch.cat([local_trajs, trajs[..., 5:]], dim=-1)
    else:
        trajs = local_trajs

    return trajs


def batch_transform_polylines_to_local_frame(polylines):
    """
    Batch transform polylines to the local frame of reference.

    Args:
        polylines (torch.Tensor): Polylines tensor of shape [B, M, W, 5].

    Returns:
        torch.Tensor: Transformed polylines in the local frame.

    """
    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]

    local_x = (x - x[:, :, 0, None]) * torch.cos(theta[:, :, 0, None]) + (
        y - y[:, :, 0, None]
    ) * torch.sin(theta[:, :, 0, None])
    local_y = -(x - x[:, :, 0, None]) * torch.sin(theta[:, :, 0, None]) + (
        y - y[:, :, 0, None]
    ) * torch.cos(theta[:, :, 0, None])

    local_theta = theta - theta[:, :, 0, None]
    local_theta = wrap_angle(local_theta)

    local_polylines = torch.stack([local_x, local_y, local_theta], dim=-1)
    local_polylines[polylines[..., :3] == 0] = 0
    polylines = torch.cat([local_polylines, polylines[..., 3:]], dim=-1)

    return polylines


def batch_transform_trajs_to_global_frame(trajs, current_states):
    """
    Batch transform trajectories to the global frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, x, 2 or 3].
        current_states (torch.Tensor): Current states tensor of shape [B, N, 5].

    Returns:
        torch.Tensor: Transformed trajectories in the global frame. [B, N, x, 3]

    """
    x, y, theta = (
        current_states[:, :, 0],
        current_states[:, :, 1],
        current_states[:, :, 2],
    )
    g_x = trajs[..., 0] * torch.cos(theta[:, :, None]) - trajs[
        ..., 1
    ] * torch.sin(theta[:, :, None])
    g_y = trajs[..., 0] * torch.sin(theta[:, :, None]) + trajs[
        ..., 1
    ] * torch.cos(theta[:, :, None])
    x = g_x + x[:, :, None]
    y = g_y + y[:, :, None]

    if trajs.shape[-1] == 2:
        trajs = torch.stack([x, y], dim=-1)
    else:
        theta = trajs[..., 2] + theta[:, :, None]
        theta = wrap_angle(theta)
        trajs = torch.stack([x, y, theta], dim=-1)

    return trajs


def wrap_angle(angle):
    """
    Wrap the angle to [-pi, pi].

    Args:
        angle (torch.Tensor): Angle tensor.

    Returns:
        torch.Tensor: Wrapped angle.

    """
    # return torch.atan2(torch.sin(angle), torch.cos(angle))
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def inverse_kinematics(
    agents_future: torch.Tensor,
    agents_future_valid: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 5,
):
    """
    Perform inverse kinematics to compute actions.

    Args:
        agents_future (torch.Tensor): Future agent positions tensor.
            [B, A, T, 8] # x, y, yaw, velx, vely, length, width, height
        agents_future_valid (torch.Tensor): Future agent validity tensor. [B, A, T]
        dt (float): Time interval. Default is 0.1.
        action_len (int): Length of each action. Default is 5.

    Returns:
        torch.Tensor: Predicted actions.

    """
    # Inverse kinematics implementation goes here
    batch_size, num_agents, num_timesteps, _ = agents_future.shape
    assert (
        num_timesteps - 1
    ) % action_len == 0, "future_len must be divisible by action_len"
    num_actions = (num_timesteps - 1) // action_len

    yaw = agents_future[..., 2]
    speed = torch.norm(agents_future[..., 3:5], dim=-1)

    yaw_rate = wrap_angle(torch.diff(yaw, dim=-1)) / dt
    accel = torch.diff(speed, dim=-1) / dt
    action_valid = agents_future_valid[..., :1] & agents_future_valid[..., 1:]

    # filter out invalid actions
    yaw_rate = torch.where(action_valid, yaw_rate, 0.0)
    accel = torch.where(action_valid, accel, 0.0)

    # Reshape for mean pooling
    yaw_rate = yaw_rate.reshape(batch_size, num_agents, num_actions, -1)
    accel = accel.reshape(batch_size, num_agents, num_actions, -1)
    action_valid = action_valid.reshape(
        batch_size, num_agents, num_actions, -1
    )

    yaw_rate_sample = yaw_rate.sum(dim=-1) / torch.clamp(
        action_valid.sum(dim=-1), min=1.0
    )
    accel_sample = accel.sum(dim=-1) / torch.clamp(
        action_valid.sum(dim=-1), min=1.0
    )
    action = torch.stack([accel_sample, yaw_rate_sample], dim=-1)
    action_valid = action_valid.any(dim=-1)

    # Filter again
    action = torch.where(action_valid[..., None], action, 0.0)

    return action, action_valid


def roll_out(
    current_states: torch.Tensor,
    actions: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 5,
    global_frame: float = True,
):
    """
    Forward pass of the dynamics model.

    Args:
        current_states (torch.Tensor): Current states tensor of shape [B, N, x, 5]. [x, y, theta, v_x, v_y]
        actions (torch.Tensor): Inputs tensor of shape [B, N, x, T_f//T_a, 2]. [Accel, yaw_rate]
        global_frame (bool): Flag indicating whether to use the global frame of reference. Default is False.

    Returns:
        torch.Tensor: Predicted trajectories.

    """
    x = current_states[..., 0]
    y = current_states[..., 1]
    theta = current_states[..., 2]
    v_x = current_states[..., 3]
    v_y = current_states[..., 4]
    v = torch.sqrt(v_x**2 + v_y**2)

    a = actions[..., 0].repeat_interleave(action_len, dim=-1)
    v = v.unsqueeze(-1) + torch.cumsum(a * dt, dim=-1)
    v += torch.randn_like(v) * 0.1
    v = torch.clamp(v, min=0)

    yaw_rate = actions[..., 1].repeat_interleave(action_len, dim=-1)
    yaw_rate += torch.randn_like(yaw_rate) * 0.01

    if global_frame:
        theta = theta.unsqueeze(-1) + torch.cumsum(yaw_rate * dt, dim=-1)
    else:
        theta = torch.cumsum(yaw_rate * dt, dim=2)

    # theta = torch.fmod(theta + torch.pi, 2*torch.pi) - torch.pi
    # theta = wrap_angle(theta)

    v_x = v * torch.cos(theta)
    v_y = v * torch.sin(theta)

    if global_frame:
        x = x.unsqueeze(-1) + torch.cumsum(v_x * dt, dim=-1)
        y = y.unsqueeze(-1) + torch.cumsum(v_y * dt, dim=-1)
    else:
        x = torch.cumsum(v_x * dt, dim=-1)
        y = torch.cumsum(v_y * dt, dim=-1)

    return torch.stack([x, y, theta, v_x, v_y], dim=-1)
