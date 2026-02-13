"""Trajectory smoothing utilities (Torch-only, no SciPy dependency).

Designed for short-horizon predicted trajectories where we want a smooth and
kinematically consistent (x, y, yaw, v) sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    two_pi = 2.0 * torch.pi
    return torch.remainder(angle + torch.pi, two_pi) - torch.pi


def unwrap_yaw(yaw: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unwrap a yaw angle sequence along `dim` (Torch equivalent of np.unwrap).

    Args:
        yaw: Tensor containing angles in radians.
        dim: Dimension that corresponds to time.
    """
    # Move dim to last for simplicity
    if dim != -1:
        yaw = yaw.movedim(dim, -1)

    if yaw.shape[-1] <= 1:
        out = yaw
    else:
        dyaw = _wrap_to_pi(yaw[..., 1:] - yaw[..., :-1])
        out = torch.cat([yaw[..., :1], yaw[..., :1] + torch.cumsum(dyaw, dim=-1)], dim=-1)

    if dim != -1:
        out = out.movedim(-1, dim)
    return out


def symmetric_moving_average(x: torch.Tensor, window: int, dim: int = -1) -> torch.Tensor:
    """Non-causal moving average with replicate padding along `dim`."""
    if window <= 1:
        return x
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")

    # Move dim to last and reshape to (N, C, L) for avg_pool1d
    if dim != -1:
        x = x.movedim(dim, -1)
    *prefix, L = x.shape
    x_ = x.reshape(-1, 1, L)

    pad = window // 2
    x_pad = F.pad(x_, (pad, pad), mode="replicate")
    y = F.avg_pool1d(x_pad, kernel_size=window, stride=1)
    y = y.reshape(*prefix, L)

    if dim != -1:
        y = y.movedim(-1, dim)
    return y


def finite_difference(x: torch.Tensor, dt: float, dim: int = -1) -> torch.Tensor:
    """Compute dx/dt along `dim` with central differences (forward/backward at ends)."""
    if dim != -1:
        x = x.movedim(dim, -1)

    L = x.shape[-1]
    if L == 0:
        out = x
    elif L == 1:
        out = torch.zeros_like(x)
    else:
        out = torch.empty_like(x)
        out[..., 0] = (x[..., 1] - x[..., 0]) / dt
        out[..., -1] = (x[..., -1] - x[..., -2]) / dt
        if L > 2:
            out[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / (2.0 * dt)

    if dim != -1:
        out = out.movedim(-1, dim)
    return out


def smooth_xy_yaw_v(
    traj_xy_yaw_v: torch.Tensor,
    *,
    dt: float = 0.1,
    window: int = 7,
    yaw_blend_from_xy: float = 0.7,
    speed_eps: float = 0.2,
) -> torch.Tensor:
    """Smooth (x, y, yaw, v) and enforce consistency between them.

    This is an *offline* (non-causal) smoother intended for visualization or
    trajectory post-processing, not for real-time closed-loop control.

    Strategy:
    - Smooth x,y with symmetric moving average.
    - Unwrap yaw, smooth it, wrap back.
    - Recompute vx,vy from smoothed x,y and speed v = hypot(vx,vy).
    - Compute yaw_from_xy = atan2(vy, vx), then blend with smoothed yaw when speed is reliable.

    Args:
        traj_xy_yaw_v: [..., T, 4] with last dim = (x, y, yaw, v).
        dt: timestep (seconds).
        window: odd MA window size.
        yaw_blend_from_xy: how much to trust yaw derived from (x,y) derivatives when speed > speed_eps.
        speed_eps: below this speed, keep yaw mostly from smoothed yaw (derivative yaw is noisy).
    """
    if traj_xy_yaw_v.shape[-1] != 4:
        raise ValueError(f"Expected last dim=4 (x,y,yaw,v), got {traj_xy_yaw_v.shape[-1]}")

    x = traj_xy_yaw_v[..., 0]
    y = traj_xy_yaw_v[..., 1]
    yaw = traj_xy_yaw_v[..., 2]

    x_s = symmetric_moving_average(x, window=window, dim=-1)
    y_s = symmetric_moving_average(y, window=window, dim=-1)

    yaw_u = unwrap_yaw(yaw, dim=-1)
    yaw_s = symmetric_moving_average(yaw_u, window=window, dim=-1)
    yaw_s = _wrap_to_pi(yaw_s)

    vx = finite_difference(x_s, dt=dt, dim=-1)
    vy = finite_difference(y_s, dt=dt, dim=-1)
    v = torch.sqrt(vx * vx + vy * vy).clamp_min(0.0)

    yaw_from_xy = torch.atan2(vy, vx)

    # Blend yaw on the unit circle to avoid wrap issues
    a = float(yaw_blend_from_xy)
    trust = (v > speed_eps).to(yaw_s.dtype)  # [..., T]
    a_eff = a * trust

    c = (1.0 - a_eff) * torch.cos(yaw_s) + a_eff * torch.cos(yaw_from_xy)
    s = (1.0 - a_eff) * torch.sin(yaw_s) + a_eff * torch.sin(yaw_from_xy)
    yaw_out = torch.atan2(s, c)

    out = torch.stack([x_s, y_s, yaw_out, v], dim=-1)
    return out


def smooth_predicted_trajectories_xy_yaw_speed(
    predicted_trajectories: torch.Tensor,
    *,
    dt: float = 0.1,
    window: int = 7,
    yaw_index: int = 2,
    speed_index: int = 5,
    yaw_blend_from_xy: float = 0.7,
    speed_eps: float = 0.2,
) -> torch.Tensor:
    """Smooth a predicted trajectory tensor that has at least x,y and optionally yaw/speed.

    Expected layout: [..., T, D] where x=0, y=1. If D has yaw/speed at the
    provided indices, they will be replaced with smoothed values. Extra dims
    are preserved.
    """
    if predicted_trajectories.shape[-1] < 2:
        raise ValueError("predicted_trajectories must have at least (x,y) in last dim")

    D = predicted_trajectories.shape[-1]
    out = predicted_trajectories.clone()

    x = out[..., 0]
    y = out[..., 1]
    yaw = out[..., yaw_index] if yaw_index < D else torch.zeros_like(x)
    v = out[..., speed_index] if speed_index < D else torch.zeros_like(x)

    packed = torch.stack([x, y, yaw, v], dim=-1)
    packed_s = smooth_xy_yaw_v(
        packed,
        dt=dt,
        window=window,
        yaw_blend_from_xy=yaw_blend_from_xy,
        speed_eps=speed_eps,
    )

    out[..., 0] = packed_s[..., 0]
    out[..., 1] = packed_s[..., 1]
    if yaw_index < D:
        out[..., yaw_index] = packed_s[..., 2]
    if speed_index < D:
        out[..., speed_index] = packed_s[..., 3]
    return out

