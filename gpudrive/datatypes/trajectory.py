import torch
from dataclasses import dataclass
import madrona_gpudrive

TRAJ_LEN = 91  # Length of the logged trajectory


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
        self.yaw = raw_logs[:, :, 4 * TRAJ_LEN: 5 * TRAJ_LEN].view(
            num_worlds, max_agents, TRAJ_LEN, -1
        )
        self.valids = raw_logs[:, :, 5 * TRAJ_LEN:6 * TRAJ_LEN].view(
            num_worlds, max_agents, TRAJ_LEN, -1
        ).to(torch.int32)
        self.inferred_actions = raw_logs[:, :, 6 * TRAJ_LEN: 16 * TRAJ_LEN].view(
            num_worlds, max_agents, TRAJ_LEN, -1
        )

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
                expert_traj_tensor.to_torch().clone().to(device), num_worlds, max_agents
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

@dataclass
class VBDTrajectory:
    """A class to represent the VBD predicted trajectories.
    Initialized from `vbd_trajectory_tensor` (src/bindings.cpp).
    Shape: (num_worlds, max_agents, traj_len, 5)

    Attributes:
        trajectories: Tensor of shape (num_worlds, max_agents, traj_len, 5) containing
            position (x, y), heading (yaw), and velocity (vx, vy) for each timestep.
    """

    def __init__(
        self, vbd_traj_tensor: torch.Tensor
    ):
        """Initializes the VBD trajectory with a tensor."""
        self.pos_x = vbd_traj_tensor[:, :, :, 0]
        self.pos_y = vbd_traj_tensor[:, :, :, 1]
        self.pos_xy = torch.stack([self.pos_x, self.pos_y], dim=3)
        self.yaw = vbd_traj_tensor[:, :, :, 2]
        self.vel_x = vbd_traj_tensor[:, :, :, 3]
        self.vel_y = vbd_traj_tensor[:, :, :, 4] 

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

    # def restore_mean(self, mean_x, mean_y):
    #     """Reapplies the mean to revert back to the original coordinates."""
    #     # Reshape for broadcasting
    #     mean_x_reshaped = mean_x.view(-1, 1, 1)
    #     mean_y_reshaped = mean_y.view(-1, 1, 1)
        
    #     # Apply to x and y coordinates
    #     self.trajectories[..., 0] += mean_x_reshaped
    #     self.trajectories[..., 1] += mean_y_reshaped