import torch
from dataclasses import dataclass
import madrona_gpudrive

TRAJ_LEN = 91  # Length of the logged trajectory


def wrap_yaws(agent_headings: torch.Tensor) -> torch.Tensor:
    """
    Wrap an array of angles to the range [-pi, pi]
    """
    return ((agent_headings + torch.pi) % (2 * torch.pi)) - torch.pi


def to_local_frame(
    global_pos_xy: torch.Tensor,
    ego_pos: torch.Tensor,
    ego_yaw: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Transform trajectory from global coordinates to ego-centric frame.
    Args:
        global_pos_xy: Shape (time_steps, 2) containing x,y coordinates in global frame
        ego_pos: Shape (2,) containing ego x,y position
        ego_yaw: Shape (1,) containing ego yaw angle in radians
    Returns:
        transformed_trajectory: Shape (time_steps, 2) in ego-centric frame
    """
    # Step 1: Translate trajectory to be relative to ego position
    translated = global_pos_xy - ego_pos

    # Step 2: Rotate trajectory to align with ego orientation
    # Create rotation matrix
    cos_yaw = torch.cos(ego_yaw).to(device)
    sin_yaw = torch.sin(ego_yaw).to(device)
    rotation_matrix = torch.tensor(
        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]]
    ).to(device)

    # Apply rotation matrix to the translated trajectory
    # We need to transpose rotation_matrix for batch matrix multiplication
    transformed_trajectory = torch.matmul(translated, rotation_matrix.T)

    return transformed_trajectory


@dataclass
class LogTrajectory:
    """A class to represent the logged human trajectories. Initialized from `expert_trajectory_tensor` (src/bindings.cpp).
    Shape: (num_worlds, max_agents, TRAJ_LEN, action_space)

    Attributes:
        pos_xy: Global (but demeaned) positions of the agent(s) across the trajectory.
        vel_xy: Global (but demeaned) velocity of the agent(s) across the trajectory.
        yaw: Headings (yaw angles) of the agent(s) across the trajectory.
        valids: Valid flag for each timestep in the trajectory.
        actions: Expert actions performed by the agent(s) across the trajectory.
    """

    def __init__(
        self, raw_logs: torch.Tensor, num_worlds: int, max_agents: int
    ):
        """Initializes the expert trajectory with an observation tensor."""
        self.pos_xy = raw_logs[:, :, : 2 * TRAJ_LEN].view(
            num_worlds, max_agents, TRAJ_LEN, -1
        )
        self.vel_xy = raw_logs[:, :, 2 * TRAJ_LEN : 4 * TRAJ_LEN].view(
            num_worlds, max_agents, TRAJ_LEN, -1
        )
        self.yaw = raw_logs[:, :, 4 * TRAJ_LEN : 5 * TRAJ_LEN].view(
            num_worlds, max_agents, TRAJ_LEN, -1
        )
        self.valids = (
            raw_logs[:, :, 5 * TRAJ_LEN : 6 * TRAJ_LEN]
            .view(num_worlds, max_agents, TRAJ_LEN, -1)
            .to(torch.int32)
        )
        self.inferred_actions = raw_logs[
            :, :, 6 * TRAJ_LEN : 16 * TRAJ_LEN
        ].view(num_worlds, max_agents, TRAJ_LEN, -1)

        # Zero-out invalid timesteps
        self.pos_xy = self.pos_xy * self.valids
        self.vel_xy = self.vel_xy * self.valids
        self.yaw = self.yaw * self.valids
        self.ref_speed = self.comp_reference_speed()

    @classmethod
    def from_tensor(
        cls,
        expert_traj_tensor: madrona_gpudrive.madrona.Tensor,
        num_worlds: int,
        max_agents: int,
        backend="torch",
        device="cuda",
    ):
        """Creates an LogTrajectory from a tensor."""
        if backend == "torch":
            return cls(
                expert_traj_tensor.to_torch().clone().to(device),
                num_worlds,
                max_agents,
            )  # Pass the entire tensor
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def restore_mean(self, mean_x, mean_y):
        """Reapplies the mean to revert back to the original coordinates."""
        # Reshape for broadcasting: [num_worlds, agents, time_steps]
        mean_x_reshaped = mean_x.view(-1, 1, 1)
        mean_y_reshaped = mean_y.view(-1, 1, 1)

        # Apply to x and y coordinates
        self.pos_xy[:, :, :, 0] += mean_x_reshaped
        self.pos_xy[:, :, :, 1] += mean_y_reshaped

    def comp_reference_speed(self):
        """Returns the average speed of the trajectory."""
        return torch.sqrt(
            self.vel_xy[:, :, :, 0] ** 2 + self.vel_xy[:, :, :, 1] ** 2
        ).unsqueeze(-1)

    @property
    def length(self):
        """Returns the length of the trajectory."""
        return self.pos_xy.shape[2]


@dataclass
class VBDTrajectoryOnline:
    """A class to represent the VBD predicted trajectories.
    Initialized from VBD predictions.
    Shape: (num_worlds, max_agents, traj_len, 5)

    Attributes:
        trajectories: Tensor of shape (num_worlds, max_agents, traj_len, 5) containing
            position (x, y), heading (yaw), and velocity (vx, vy) for each timestep.
    """

    def __init__(
        self, vbd_traj_tensor: torch.Tensor, mean_pos_xy: torch.Tensor
    ):
        """Initializes the VBD trajectory with a tensor."""
        self.pos_x = vbd_traj_tensor[:, :, :, 0].unsqueeze(-1)
        self.pos_y = vbd_traj_tensor[:, :, :, 1].unsqueeze(-1)
        self.pos_xy = vbd_traj_tensor[:, :, :, :2]
        self.yaw = wrap_yaws(vbd_traj_tensor[:, :, :, 2].unsqueeze(-1))
        self.vel_x = vbd_traj_tensor[:, :, :, 3].unsqueeze(-1)
        self.vel_y = vbd_traj_tensor[:, :, :, 4].unsqueeze(-1)
        self.vel_xy = vbd_traj_tensor[:, :, :, 3:5]
        self.ref_speed = self.comp_reference_speed()
        # Assumption: All timesteps are valid (correct)
        self.valids = vbd_traj_tensor[:, :, :, 5].unsqueeze(-1).to(
            torch.int32
        )

        self.mean_pos_xy = mean_pos_xy
        self.mean_x = mean_pos_xy[:, 0]
        self.mean_y = mean_pos_xy[:, 1]

    @classmethod
    def from_tensor(
        cls,
        vbd_predictions,
        mean_pos_xy,
        backend="torch",
        device="cuda",
    ):
        """Creates a VBDTrajectory from a tensor."""
        if backend == "torch":
            return cls(
                vbd_predictions.clone().to(device),
                mean_pos_xy.clone().to(device),
            )
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def comp_reference_speed(self):
        """Returns the average speed of the trajectory."""
        return torch.sqrt(self.vel_x**2 + self.vel_y**2)

    @property
    def length(self):
        """Returns the length of the trajectory."""
        return self.pos_xy.shape[2]

    def demean_positions(self):
        """
        In GPUDrive, we center everything at zero. By default, the VBD
        predictions are not centered at zero, so we demean the predicted trajectory.
        """
        # Reshape for broadcasting
        mean_x_reshaped = self.mean_x.view(-1, 1, 1)
        mean_y_reshaped = self.mean_y.view(-1, 1, 1)

        # Apply to x and y coordinates
        self.pos_xy[..., 10:, 0] -= mean_x_reshaped
        self.pos_xy[..., 10:, 1] -= mean_y_reshaped

        self.pos_x = self.pos_xy[..., 0].unsqueeze(-1)
        self.pos_y = self.pos_xy[..., 1].unsqueeze(-1)


@dataclass
class VBDTrajectory:
    """A class to represent the VBD predicted trajectories.
    Initialized from `vbd_trajectory_tensor` (src/bindings.cpp).
    Shape: (num_worlds, max_agents, traj_len, 5)
            where traj_len is 90 - initialization steps.

    Attributes:
        trajectories: Tensor of shape (num_worlds, max_agents, traj_len, 5) containing
            position (x, y), heading (yaw), and velocity (vx, vy) for each timestep.
    """

    def __init__(self, vbd_traj_tensor: torch.Tensor):
        """Initializes the VBD trajectory with a tensor."""
        # Pad to make sure the trajectory is of length 91
        padding = (0, 0, 1, 0)

        vbd_traj_tensor_pad = torch.nn.functional.pad(vbd_traj_tensor, padding, mode='constant', value=-1.0)

        self.pos_x = vbd_traj_tensor_pad[:, :, :, 0].unsqueeze(-1)
        self.pos_y = vbd_traj_tensor_pad[:, :, :, 1].unsqueeze(-1)
        self.pos_xy = vbd_traj_tensor_pad[:, :, :, :2]
        self.yaw = wrap_yaws(vbd_traj_tensor_pad[:, :, :, 2].unsqueeze(-1))
        self.vel_x = vbd_traj_tensor_pad[:, :, :, 3].unsqueeze(-1)
        self.vel_y = vbd_traj_tensor_pad[:, :, :, 4].unsqueeze(-1)
        self.vel_xy = vbd_traj_tensor_pad[:, :, :, 3:5]
        self.ref_speed = self.comp_reference_speed()
        self.valids = vbd_traj_tensor_pad[:, :, :, 5].unsqueeze(-1).to(
            torch.int32
        )
        
    @classmethod
    def from_tensor(
        cls,
        vbd_traj_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        """Creates a VBDTrajectory from a tensor."""
        if backend == "torch":
            return cls(vbd_traj_tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    def comp_reference_speed(self):
        """Returns the average speed of the trajectory."""
        return torch.sqrt(self.vel_x**2 + self.vel_y**2)

    @property
    def length(self):
        """Returns the length of the trajectory."""
        return self.pos_xy.shape[2]
