"""SQP (Sequential Quadratic Programming) trajectory smoothing.

Uses scipy SLSQP to produce smooth, kinematically consistent trajectories
by jointly optimising (x, y, yaw, speed).

Problem formulation (4-D)
-------------------------
Decision variables  z = [x₀…x_{N-1}, y₀…y_{N-1}, ψ₀…ψ_{N-1}, v₀…v_{N-1}]

    min   w_pc ‖D₂x‖² + w_pc ‖D₂y‖²          (position curvature)
        + w_pj ‖D₃x‖² + w_pj ‖D₃y‖²          (position jerk)
        + w_yr ‖D₁ψ‖²                          (yaw-rate smoothness)
        + w_ya ‖D₂ψ‖²                          (yaw acceleration)
        + w_sa ‖D₁v‖²                          (speed acceleration)
        + w_sj ‖D₂v‖²                          (speed jerk)
        + w_kin Σ[(Δxᵢ − vᵢcosψᵢ·dt)²
                + (Δyᵢ − vᵢsinψᵢ·dt)²]        (kinematic consistency)
        + w_dxy ‖p_xy − p_xy⁰‖²                (xy fidelity)
        + w_dψ  ‖ψ − ψ⁰‖²                     (yaw fidelity)
        + w_dv  ‖v − v⁰‖²                      (speed fidelity)

    s.t.  box bounds on every variable
          fixed endpoints (optional)

All gradients are provided analytically for fast convergence.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Finite difference matrices
# ---------------------------------------------------------------------------

def _build_diff_matrix(n: int, order: int) -> np.ndarray:
    """``(n - order) × n`` finite-difference matrix of given *order*."""
    if order == 1:
        D = np.zeros((n - 1, n))
        for i in range(n - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
    elif order == 2:
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
    elif order == 3:
        D = np.zeros((n - 3, n))
        for i in range(n - 3):
            D[i, i] = -1.0
            D[i, i + 1] = 3.0
            D[i, i + 2] = -3.0
            D[i, i + 3] = 1.0
    else:
        raise ValueError(f"Unsupported order: {order}")
    return D


# ---------------------------------------------------------------------------
# 2-D smoother  (x, y only — kept for backward compat)
# ---------------------------------------------------------------------------

def sqp_smooth_trajectory(
    points: np.ndarray,
    *,
    w_curvature: float = 10.0,
    w_jerk: float = 5.0,
    w_deviation: float = 1.0,
    max_deviation: float = 2.0,
    fix_endpoints: bool = True,
    max_curvature: Optional[float] = None,
    max_iter: int = 200,
) -> np.ndarray:
    """Smooth a 2-D trajectory ``(x, y)`` via SQP (SLSQP).

    See module docstring for the general idea; this variant only touches
    spatial coordinates and ignores heading / speed.

    Parameters
    ----------
    points : (N, 2)
        Original waypoints ``[x, y]``.
    w_curvature / w_jerk / w_deviation :
        Cost weights for curvature, jerk, and deviation.
    max_deviation :
        Per-axis box bound (metres).
    fix_endpoints :
        Pin first / last waypoints.
    max_curvature :
        Optional upper bound on discrete curvature (m⁻¹).
    max_iter :
        SLSQP iteration limit.

    Returns
    -------
    smoothed : (N, 2)
    """
    N = len(points)
    if N < 4:
        return points.copy()

    px = points[:, 0].copy()
    py = points[:, 1].copy()
    z0 = np.concatenate([px, py])
    p_orig = z0.copy()

    D2 = _build_diff_matrix(N, 2)
    H_single = w_curvature * (D2.T @ D2) + w_deviation * np.eye(N)

    D3 = None
    if N >= 4 and w_jerk > 0:
        D3 = _build_diff_matrix(N, 3)
        H_single = H_single + w_jerk * (D3.T @ D3)

    def objective(z: np.ndarray) -> float:
        x, y = z[:N], z[N:]
        dx, dy = x - px, y - py
        d2x, d2y = D2 @ x, D2 @ y
        cost = w_curvature * (d2x @ d2x + d2y @ d2y) + w_deviation * (dx @ dx + dy @ dy)
        if D3 is not None:
            d3x, d3y = D3 @ x, D3 @ y
            cost += w_jerk * (d3x @ d3x + d3y @ d3y)
        return cost

    def gradient(z: np.ndarray) -> np.ndarray:
        x, y = z[:N], z[N:]
        gx = 2.0 * (H_single @ x) - 2.0 * w_deviation * px
        gy = 2.0 * (H_single @ y) - 2.0 * w_deviation * py
        return np.concatenate([gx, gy])

    lb = np.full(2 * N, -np.inf)
    ub = np.full(2 * N, np.inf)
    for i in range(N):
        if fix_endpoints and (i == 0 or i == N - 1):
            lb[i] = ub[i] = px[i]
            lb[N + i] = ub[N + i] = py[i]
        else:
            lb[i] = px[i] - max_deviation
            ub[i] = px[i] + max_deviation
            lb[N + i] = py[i] - max_deviation
            ub[N + i] = py[i] + max_deviation

    constraints: list = []
    if max_curvature is not None:
        for i in range(1, N - 1):
            def _mk(idx: int):
                def con(z: np.ndarray) -> float:
                    x, y = z[:N], z[N:]
                    dx1, dy1 = x[idx] - x[idx - 1], y[idx] - y[idx - 1]
                    dx2, dy2 = x[idx + 1] - x[idx], y[idx + 1] - y[idx]
                    ds_sq = dx1 ** 2 + dy1 ** 2 + 1e-12
                    return max_curvature * ds_sq - abs(dx1 * dy2 - dy1 * dx2)
                return con
            constraints.append({"type": "ineq", "fun": _mk(i)})

    result = minimize(
        objective, z0, method="SLSQP", jac=gradient,
        bounds=list(zip(lb, ub)), constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )
    z_opt = result.x
    return np.column_stack([z_opt[:N], z_opt[N:]])


# ---------------------------------------------------------------------------
# 4-D smoother  (x, y, yaw, speed) — kinematically consistent
# ---------------------------------------------------------------------------

def sqp_smooth_trajectory_xyav(
    states: np.ndarray,
    *,
    dt: float = 0.1,
    w_pos_curv: float = 10.0,
    w_pos_jerk: float = 5.0,
    w_yaw_rate: float = 8.0,
    w_yaw_accel: float = 3.0,
    w_speed_accel: float = 8.0,
    w_speed_jerk: float = 3.0,
    w_kinematic: float = 15.0,
    w_deviation_xy: float = 1.0,
    w_deviation_yaw: float = 2.0,
    w_deviation_speed: float = 2.0,
    max_deviation_xy: float = 2.0,
    max_deviation_yaw: float = 0.3,
    max_deviation_speed: float = 3.0,
    fix_endpoints: bool = True,
    obstacle_points: Optional[np.ndarray] = None,
    w_obstacle: float = 0.0,
    obstacle_safe_radius: float = 3.0,
    obstacle_end_weight: float = 1.0,
    max_iter: int = 300,
) -> np.ndarray:
    """Smooth a 4-D kinematic trajectory ``(x, y, ψ, v)`` via SQP.

    The optimiser enforces *kinematic consistency* between position,
    heading, and speed via a soft penalty that couples all four channels.
    This produces trajectories that are not only spatially smooth but
    also physically plausible for a bicycle-kinematic vehicle.

    Parameters
    ----------
    states : (N, 4)
        ``[x, y, yaw, speed]`` per waypoint.
    dt :
        Time step between consecutive waypoints (seconds).
    w_pos_curv / w_pos_jerk :
        Position 2nd / 3rd-order smoothness weights.
    w_yaw_rate / w_yaw_accel :
        Heading 1st / 2nd-order smoothness weights.
    w_speed_accel / w_speed_jerk :
        Speed 1st / 2nd-order smoothness weights.
    w_kinematic :
        Weight on the kinematic consistency penalty:
        ``Σ (Δx − v·cos ψ·dt)² + (Δy − v·sin ψ·dt)²``.
    w_deviation_xy / w_deviation_yaw / w_deviation_speed :
        Fidelity weights for each channel.
    max_deviation_xy / max_deviation_yaw / max_deviation_speed :
        Per-axis box bounds on each channel.
    obstacle_points : (M, 2), optional
        障碍物点云（世界坐标），用于对轨迹施加软避障代价。
    w_obstacle :
        障碍物软惩罚权重（0 表示关闭）。
    obstacle_safe_radius :
        安全半径，轨迹点进入该半径后会被惩罚。
    obstacle_end_weight :
        时间加权终点系数（>1 时后段避障更强）。
    fix_endpoints :
        Pin first / last waypoints.
    max_iter :
        SLSQP iteration limit.

    Returns
    -------
    smoothed : (N, 4)
        ``[x, y, yaw, speed]`` — smoothed and kinematically consistent.
    """
    N = len(states)
    if N < 4:
        return states.copy()

    # ---- Original values ----
    x0 = states[:, 0].copy()
    y0 = states[:, 1].copy()
    yaw0 = np.unwrap(states[:, 2].copy())     # unwrap for meaningful diffs
    v0 = states[:, 3].copy()

    z_init = np.concatenate([x0, y0, yaw0, v0])

    # ---- Finite difference matrices ----
    D1 = _build_diff_matrix(N, 1)              # (N-1, N)
    D2 = _build_diff_matrix(N, 2)              # (N-2, N)
    D3 = _build_diff_matrix(N, 3) if N >= 4 else None  # (N-3, N)

    # ---- Per-channel quadratic Hessians (including deviation) ----
    H_xy = w_pos_curv * (D2.T @ D2) + w_deviation_xy * np.eye(N)
    if D3 is not None and w_pos_jerk > 0:
        H_xy = H_xy + w_pos_jerk * (D3.T @ D3)

    H_psi = (w_yaw_rate * (D1.T @ D1)
             + w_yaw_accel * (D2.T @ D2)
             + w_deviation_yaw * np.eye(N))

    H_v = w_speed_accel * (D1.T @ D1) + w_deviation_speed * np.eye(N)
    if D3 is not None and w_speed_jerk > 0:
        H_v = H_v + w_speed_jerk * (D2.T @ D2)

    # ---- Obstacle preprocessing ----
    obs_xy: Optional[np.ndarray] = None
    obs_time_weights = np.ones(N, dtype=np.float64)
    if obstacle_end_weight != 1.0:
        obs_time_weights = np.linspace(1.0, obstacle_end_weight, N, dtype=np.float64)

    if obstacle_points is not None and w_obstacle > 0.0 and obstacle_safe_radius > 0.0:
        obs_xy_raw = np.asarray(obstacle_points, dtype=np.float64).reshape(-1, 2)
        if obs_xy_raw.size > 0:
            finite_mask = np.isfinite(obs_xy_raw).all(axis=1)
            obs_xy_raw = obs_xy_raw[finite_mask]
            if obs_xy_raw.shape[0] > 0:
                obs_xy = obs_xy_raw

    # ---- Objective ----
    def objective(z: np.ndarray) -> float:
        x  = z[0*N : 1*N]
        y  = z[1*N : 2*N]
        psi = z[2*N : 3*N]
        v  = z[3*N : 4*N]

        # Position smoothness
        d2x, d2y = D2 @ x, D2 @ y
        cost = w_pos_curv * (d2x @ d2x + d2y @ d2y)
        if D3 is not None and w_pos_jerk > 0:
            d3x, d3y = D3 @ x, D3 @ y
            cost += w_pos_jerk * (d3x @ d3x + d3y @ d3y)

        # Yaw smoothness
        d1psi = D1 @ psi
        d2psi = D2 @ psi
        cost += w_yaw_rate * (d1psi @ d1psi) + w_yaw_accel * (d2psi @ d2psi)

        # Speed smoothness
        d1v = D1 @ v
        cost += w_speed_accel * (d1v @ d1v)
        if D3 is not None and w_speed_jerk > 0:
            d2v = D2 @ v
            cost += w_speed_jerk * (d2v @ d2v)

        # Kinematic consistency  (bicycle forward model)
        cos_psi = np.cos(psi[:-1])
        sin_psi = np.sin(psi[:-1])
        v_h = v[:-1]
        ex = (x[1:] - x[:-1]) - v_h * cos_psi * dt
        ey = (y[1:] - y[:-1]) - v_h * sin_psi * dt
        cost += w_kinematic * (ex @ ex + ey @ ey)

        # Obstacle clearance penalty (soft hinge)
        if obs_xy is not None:
            dx_obs = x[:, None] - obs_xy[None, :, 0]
            dy_obs = y[:, None] - obs_xy[None, :, 1]
            dist_obs = np.sqrt(dx_obs * dx_obs + dy_obs * dy_obs + 1e-9)
            margin = obstacle_safe_radius - dist_obs
            hinge = np.maximum(0.0, margin)
            cost += w_obstacle * np.sum(obs_time_weights[:, None] * (hinge * hinge))

        # Fidelity
        cost += (w_deviation_xy    * ((x - x0) @ (x - x0) + (y - y0) @ (y - y0))
                 + w_deviation_yaw   * ((psi - yaw0) @ (psi - yaw0))
                 + w_deviation_speed * ((v - v0) @ (v - v0)))
        return cost

    # ---- Analytical gradient ----
    def gradient(z: np.ndarray) -> np.ndarray:
        x   = z[0*N : 1*N]
        y   = z[1*N : 2*N]
        psi = z[2*N : 3*N]
        v   = z[3*N : 4*N]

        # Quadratic terms  (grad = 2·H·channel − 2·w_dev·orig)
        gx   = 2.0 * H_xy  @ x   - 2.0 * w_deviation_xy    * x0
        gy   = 2.0 * H_xy  @ y   - 2.0 * w_deviation_xy    * y0
        gpsi = 2.0 * H_psi @ psi - 2.0 * w_deviation_yaw   * yaw0
        gv   = 2.0 * H_v   @ v   - 2.0 * w_deviation_speed * v0

        # Kinematic gradient
        cos_psi = np.cos(psi[:-1])
        sin_psi = np.sin(psi[:-1])
        v_h = v[:-1]
        ex = (x[1:] - x[:-1]) - v_h * cos_psi * dt
        ey = (y[1:] - y[:-1]) - v_h * sin_psi * dt

        # ∂J_kin/∂x = 2·w_kin · D1ᵀ @ ex   (D1 is forward diff matrix)
        gx += 2.0 * w_kinematic * (D1.T @ ex)
        gy += 2.0 * w_kinematic * (D1.T @ ey)

        # ∂J_kin/∂ψ_i = 2·w_kin·dt·vᵢ·(eˣᵢ sinψᵢ − eʸᵢ cosψᵢ)  i<N-1
        gpsi[:-1] += 2.0 * w_kinematic * dt * v_h * (
            ex * sin_psi - ey * cos_psi
        )

        # ∂J_kin/∂v_i = −2·w_kin·dt·(eˣᵢ cosψᵢ + eʸᵢ sinψᵢ)      i<N-1
        gv[:-1] += -2.0 * w_kinematic * dt * (
            ex * cos_psi + ey * sin_psi
        )

        # Obstacle clearance gradient
        if obs_xy is not None:
            dx_obs = x[:, None] - obs_xy[None, :, 0]
            dy_obs = y[:, None] - obs_xy[None, :, 1]
            dist_obs = np.sqrt(dx_obs * dx_obs + dy_obs * dy_obs + 1e-9)
            margin = obstacle_safe_radius - dist_obs
            active = margin > 0.0
            if np.any(active):
                base = np.zeros_like(dist_obs)
                base[active] = -2.0 * w_obstacle * margin[active] / dist_obs[active]
                weighted = base * obs_time_weights[:, None]
                gx += np.sum(weighted * dx_obs, axis=1)
                gy += np.sum(weighted * dy_obs, axis=1)

        return np.concatenate([gx, gy, gpsi, gv])

    # ---- Box bounds ----
    lb = np.full(4 * N, -np.inf)
    ub = np.full(4 * N, np.inf)
    for i in range(N):
        if fix_endpoints and (i == 0 or i == N - 1):
            lb[0*N + i] = ub[0*N + i] = x0[i]
            lb[1*N + i] = ub[1*N + i] = y0[i]
            lb[2*N + i] = ub[2*N + i] = yaw0[i]
            lb[3*N + i] = ub[3*N + i] = v0[i]
        else:
            lb[0*N + i] = x0[i]   - max_deviation_xy
            ub[0*N + i] = x0[i]   + max_deviation_xy
            lb[1*N + i] = y0[i]   - max_deviation_xy
            ub[1*N + i] = y0[i]   + max_deviation_xy
            lb[2*N + i] = yaw0[i] - max_deviation_yaw
            ub[2*N + i] = yaw0[i] + max_deviation_yaw
            lb[3*N + i] = max(0.0, v0[i] - max_deviation_speed)
            ub[3*N + i] = v0[i]   + max_deviation_speed

    # ---- Solve ----
    result = minimize(
        objective, z_init, method="SLSQP", jac=gradient,
        bounds=list(zip(lb, ub)),
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )

    z_opt = result.x
    x_s   = z_opt[0*N : 1*N]
    y_s   = z_opt[1*N : 2*N]
    psi_s = z_opt[2*N : 3*N]
    v_s   = z_opt[3*N : 4*N]

    # Wrap yaw back to [-π, π]
    psi_s = (psi_s + np.pi) % (2.0 * np.pi) - np.pi

    return np.column_stack([x_s, y_s, psi_s, v_s])


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def smooth_trajectory_dict(
    trajectories: Dict[int, Dict[int, List[Tuple]]],
    *,
    min_points: int = 8,
    **sqp_kwargs,
) -> Dict[int, Dict[int, List[Tuple]]]:
    """Apply 4-D SQP smoothing to every trajectory in the nested dict.

    Expected format: ``{env_idx: {agent_idx: [(x, y, yaw, speed, step), …]}}``.
    """
    smoothed: Dict[int, Dict[int, List[Tuple]]] = {}
    for env_idx, agents in trajectories.items():
        smoothed[env_idx] = {}
        for agent_idx, traj in agents.items():
            if len(traj) < min_points:
                smoothed[env_idx][agent_idx] = list(traj)
                continue
            raw = np.array([[t[0], t[1], t[2], t[3]] for t in traj])
            steps = [t[4] for t in traj]
            pts_s = sqp_smooth_trajectory_xyav(raw, **sqp_kwargs)
            smoothed[env_idx][agent_idx] = [
                (pts_s[i, 0], pts_s[i, 1], pts_s[i, 2], pts_s[i, 3], steps[i])
                for i in range(len(steps))
            ]
    return smoothed
