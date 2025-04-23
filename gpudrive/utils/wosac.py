import wandb
import tensorflow as tf
import numpy as np
from waymo_open_dataset.wdl_limited.sim_agents_metrics.trajectory_features import (
    compute_displacement_error,
    compute_kinematic_features,
    compute_kinematic_validity,
)


def compute_realism_metrics(self, done_worlds):
    """Compute realism metrics.

    Args:
        done_worlds: List of indices of the worlds to track.
    """

    # [worlds, max_cont_agents]
    control_mask = (
        self.controlled_agent_mask[done_worlds].detach().cpu().numpy()
    )
    # [batch, time, 1]
    valid_mask = (
        self.env.log_trajectory.valids[done_worlds]
        .detach()
        .cpu()
        .numpy()[control_mask]
        .squeeze(-1)
    )

    # Take human logs (ground-truth)
    # Shape: [worlds, max_cont_agents, time, 2] -> [batch, time, 2]
    ref_pos_xy_np = (
        self.env.log_trajectory.pos_xy[done_worlds]
        .detach()
        .cpu()
        .numpy()[control_mask]
    )
    ref_pos_z_np = np.zeros_like(ref_pos_xy_np[:, :, 0])
    # Shape: [worlds, max_cont_agents, time, 1] -> [batch, time, 1]
    ref_headings_np = (
        self.env.log_trajectory.yaw[done_worlds]
        .detach()
        .cpu()
        .numpy()[control_mask]
        .squeeze(-1)
    )

    # Get agent information and convert to numpy
    agent_headings_np = (
        self.headings[done_worlds].detach().cpu().numpy()[control_mask]
    )
    agent_pos_xyz_np = (
        self.pos_xyz[done_worlds].detach().cpu().numpy()[control_mask]
    )

    # Extract x, y, z components
    agent_x_np = agent_pos_xyz_np[..., 0]
    agent_y_np = agent_pos_xyz_np[..., 1]
    agent_z_np = agent_pos_xyz_np[..., 2]

    ref_x_np = ref_pos_xy_np[..., 0]
    ref_y_np = ref_pos_xy_np[..., 1]

    # Convert to TensorFlow tensors
    ref_x = tf.convert_to_tensor(ref_x_np, dtype=tf.float32)
    ref_y = tf.convert_to_tensor(ref_y_np, dtype=tf.float32)
    ref_z = tf.convert_to_tensor(ref_pos_z_np, dtype=tf.float32)
    ref_heading = tf.convert_to_tensor(ref_headings_np, dtype=tf.float32)

    agent_x = tf.convert_to_tensor(agent_x_np, dtype=tf.float32)
    agent_y = tf.convert_to_tensor(agent_y_np, dtype=tf.float32)
    agent_z = tf.convert_to_tensor(agent_z_np, dtype=tf.float32)
    agent_heading = tf.convert_to_tensor(agent_headings_np, dtype=tf.float32)

    valid_mask = tf.convert_to_tensor(valid_mask, dtype=tf.bool)

    # Step duration in seconds
    seconds_per_step = 0.1  # Assuming 10Hz sampling rate

    speed_validity, accel_validity = compute_kinematic_validity(valid_mask)

    # Compute kinematic features for agents
    (
        agent_speed,
        agent_accel,
        agent_angular_speed,
        agent_angular_accel,
    ) = compute_kinematic_features(
        agent_x, agent_y, agent_z, agent_heading, seconds_per_step
    )

    # Compute kinematic features for reference trajectories
    (
        ref_speed,
        ref_accel,
        ref_angular_speed,
        ref_angular_accel,
    ) = compute_kinematic_features(
        ref_x, ref_y, ref_z, ref_heading, seconds_per_step
    )

    # Compute displacement error
    displacement_error = compute_displacement_error(
        agent_x, agent_y, agent_z, ref_x, ref_y, ref_z
    )

    # Compute additional metrics
    speed_error = tf.abs(agent_speed - ref_speed)
    accel_error = tf.abs(agent_accel - ref_accel)
    angular_speed_error = tf.abs(agent_angular_speed - ref_angular_speed)
    angular_accel_error = tf.abs(agent_angular_accel - ref_angular_accel)

    def masked_mean_no_nan_inf(tensor):
        """Compute mean excluding NaN and Inf values."""
        # Create masks for non-NaN and non-Inf values
        non_nan_mask = tf.math.logical_not(tf.math.is_nan(tensor))
        non_inf_mask = tf.math.logical_not(tf.math.is_inf(tensor))
        valid_values_mask = tf.logical_and(non_nan_mask, non_inf_mask)

        # Apply mask to tensor
        masked_tensor = tf.boolean_mask(tensor, valid_values_mask)

        # If all values are filtered out, return 0.0
        if tf.size(masked_tensor) == 0:
            return tf.constant(0.0, dtype=tf.float32)

        # Compute mean of valid values
        return tf.reduce_mean(masked_tensor)

    def masked_mean_with_validity_no_inf(tensor, validity_mask):
        """Compute mean excluding NaN and Inf values and applying validity mask."""
        # Create masks for non-NaN and non-Inf values
        non_nan_mask = tf.math.logical_not(tf.math.is_nan(tensor))
        non_inf_mask = tf.math.logical_not(tf.math.is_inf(tensor))
        data_valid_mask = tf.logical_and(non_nan_mask, non_inf_mask)

        # Combine with validity mask
        combined_mask = tf.logical_and(data_valid_mask, validity_mask)

        # Apply combined mask to tensor
        masked_tensor = tf.boolean_mask(tensor, combined_mask)

        # If all values are filtered out, return 0.0
        if tf.size(masked_tensor) == 0:
            return tf.constant(0.0, dtype=tf.float32)

        # Compute mean of valid values
        return tf.reduce_mean(masked_tensor)

    metrics = {
        "displacement_error": float(
            masked_mean_no_nan_inf(displacement_error).numpy()
        ),
        "speed_error": float(
            masked_mean_with_validity_no_inf(
                speed_error, speed_validity
            ).numpy()
        ),
        "accel_error": float(
            masked_mean_with_validity_no_inf(
                accel_error, accel_validity
            ).numpy()
        ),
        "angular_speed_error": float(
            masked_mean_with_validity_no_inf(
                angular_speed_error, speed_validity
            ).numpy()
        ),
        "angular_accel_error": float(
            masked_mean_with_validity_no_inf(
                angular_accel_error, accel_validity
            ).numpy()
        ),
        "agent_speed": float(
            masked_mean_with_validity_no_inf(
                agent_speed, speed_validity
            ).numpy()
        ),
        "agent_accel": float(
            masked_mean_with_validity_no_inf(
                agent_accel, accel_validity
            ).numpy()
        ),
        "agent_angular_speed": float(
            masked_mean_with_validity_no_inf(
                agent_angular_speed, speed_validity
            ).numpy()
        ),
        "agent_angular_accel": float(
            masked_mean_with_validity_no_inf(
                agent_angular_accel, accel_validity
            ).numpy()
        ),
        "ref_speed": float(
            masked_mean_with_validity_no_inf(ref_speed, speed_validity).numpy()
        ),
        "ref_accel": float(
            masked_mean_with_validity_no_inf(ref_accel, accel_validity).numpy()
        ),
        "ref_angular_speed": float(
            masked_mean_with_validity_no_inf(
                ref_angular_speed, speed_validity
            ).numpy()
        ),
        "ref_angular_accel": float(
            masked_mean_with_validity_no_inf(
                ref_angular_accel, accel_validity
            ).numpy()
        ),
    }

    wandb_metrics = {}
    for key, value in metrics.items():
        wandb_metrics[f"realism/{key}"] = value
    wandb.log(wandb_metrics)

    del wandb_metrics
