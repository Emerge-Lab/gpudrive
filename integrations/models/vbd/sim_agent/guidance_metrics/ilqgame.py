import jax
import jax.numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from functools import partial


@partial(jax.jit, static_argnums=(2, 3))
def dynamics(
    current_states: ArrayImpl,
    actions: ArrayImpl,
    dt: float = 0.1,
    action_len: int = 2,
):
    """
    Forward pass of the dynamics model.

    Args:
        current_states (torch.Tensor): Current states tensor of shape [..., 5]. [x, y, theta, v_x, v_y]
        actions (torch.Tensor): Inputs tensor of shape [..., 2]. [Accel, yaw_rate]

    Returns:
        torch.Tensor: Predicted next state.
    """

    # Dim： [..., 1]
    x = current_states[..., 0:1]
    y = current_states[..., 1:2]
    theta = current_states[..., 2:3]
    v_x = current_states[..., 3:4]
    v_y = current_states[..., 4:5]
    v = jnp.sqrt(v_x**2 + v_y**2)  # [..., 1]

    actions_full = jnp.expand_dims(actions, axis=-2).repeat(
        action_len, axis=-2
    )  # Dim: [..., action_len, 2]

    accel = actions_full[..., 0]
    v = v + jnp.cumsum(accel * dt, axis=-1)  # Dim: [..., action_len]
    v = jnp.clip(v, a_min=0.0, a_max=None)

    yaw_rate = actions_full[..., 1]
    yaw_rate = jnp.where(v > 0.1, yaw_rate, 0.0)
    theta = jnp.cumsum(yaw_rate * dt, axis=-1) + theta
    theta = wrap_angle(theta)
    v_x = v * jnp.cos(theta)
    v_y = v * jnp.sin(theta)

    x = jnp.cumsum(v_x * dt, axis=-1) + x
    y = jnp.cumsum(v_y * dt, axis=-1) + y
    next_states = jnp.stack([x, y, theta, v_x, v_y], axis=-1)

    return next_states[..., -1, :]


@jax.jit
def get_A_and_B(state_start, pred_action):
    jac_par = jax.jacfwd(dynamics, argnums=(0, 1))
    for _ in state_start.shape[:-1]:
        jac_par = jax.vmap(jac_par)
    return jac_par(state_start, pred_action)


def wrap_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
