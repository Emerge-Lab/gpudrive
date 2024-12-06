import gpudrive
import torch


class ControlMasks:
    """A class to represent the control type of an agent.
    Initialized from response_type_tensor.
    Shape: (num_worlds, max_controlled_agents).
    """

    def __init__(self, tensor: torch.Tensor):
        """Initializes the ego state with an observation tensor."""
        self.controlled = (tensor == 0).squeeze(-1)  # Controlled agents
        self.padding = (tensor == 1).squeeze(-1)  # Padding agents
        self.static = (tensor == 2).squeeze(-1)  # Agents marked as static
        self.valid = ~self.padding

    @classmethod
    def from_tensor(
        cls, tensor: gpudrive.madrona.Tensor, backend="torch", device="cuda"
    ):
        if backend == "torch":
            return cls(tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of the ego state tensor."""
        return self.controlled.shape
