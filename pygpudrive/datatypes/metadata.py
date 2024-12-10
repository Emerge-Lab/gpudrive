import torch
from dataclasses import dataclass
import gpudrive


class Metadata:
    """A class to represent object metadata (eg: sdc_flag)
    Initialized from metadata_tensor (src/bindings). For details, see
    `MetaData` in src/types.hpp
    
    Attributes (all masks are 0/1 int of shape (NumWorlds, NumAgents)):
        sdc_mask: Whether agent is self-driving car.
        interested: IDs of agents interested (-1 otherwise).
        prediction_mask: Whether agent's trajectory needs to be predicted (WOSAC).
        difficulty_mask: Difficulty of the agent's trajectory to be predicted (1 or 2, 0 otherwise)
    """

    def __init__(self, metadata_tensor: torch.Tensor):
        """Initializes the Metadata with the metadata tensor."""
        metadata_tensor = metadata_tensor.reshape(-1, 4, gpudrive.kMaxAgentCount)
        self.sdc_mask = metadata_tensor[:, 0, :]
        self.interested = metadata_tensor[:, 1, :]
        self.prediction_mask = metadata_tensor[:, 2, :]
        self.difficulty_mask = metadata_tensor[:, 3, :]

    @classmethod
    def from_tensor(
        cls, metadata_tensor: gpudrive.madrona.Tensor, backend="torch"
    ):
        """Creates a Metadata object from the metadata_tensor."""
        if backend == "torch":
            return cls(metadata_tensor.to_torch().clone())
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of each metadata mask."""
        return self.sdc_mask.shape