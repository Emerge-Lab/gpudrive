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
        self.inferred_actions = raw_logs[
            :, :, 6 * TRAJ_LEN : 16 * TRAJ_LEN
        ].view(num_worlds, max_agents, TRAJ_LEN, -1)

    @classmethod
    def from_tensor(
        cls,
        expert_traj_tensor: madrona_gpudrive.madrona.Tensor,
        num_worlds: int,
        max_agents: int,
        backend="torch",
    ):
        """Creates an LogTrajectory from a tensor."""
        if backend == "torch":
            return cls(
                expert_traj_tensor.to_torch().clone(), num_worlds, max_agents
            )  # Pass the entire tensor
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")
