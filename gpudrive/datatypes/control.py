import madrona_gpudrive
import torch


class ResponseType:
    """A class to represent the response type of an agent.
    Initialized from response_type_tensor.
    Shape: (num_worlds, max_controlled_agents).
    """

    def __init__(self, tensor: torch.Tensor):
        """Initializes the ego state with an observation tensor."""
        self.moving = (tensor == 0).squeeze(-1)  # Agents that are moving
        self.kinematic = (tensor == 1).squeeze(-1)  # Kinematic (not used)
        self.static = (tensor == 2).squeeze(-1)  # Static and padding agents

    @classmethod
    def from_tensor(
        cls, tensor: madrona_gpudrive.madrona.Tensor, backend="torch", device="cuda"
    ):
        if backend == "torch":
            return cls(tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of the ego state tensor."""
        return self.controlled.shape
