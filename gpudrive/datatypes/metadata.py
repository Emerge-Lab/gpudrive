import torch
from dataclasses import dataclass
import madrona_gpudrive


class Metadata:
    """A class to represent object metadata (eg: sdc_flag)
    Initialized from metadata_tensor (src/bindings). For details, see
    `MetaData` in src/types.hpp
    
    Attributes (all masks are 0/1 int of shape (NumWorlds, NumAgents)):
        id: Unique ID of the agent (non-negative int).
        isSdc: Whether agent is self-driving car (1/0, -1 padding). 
        isOfInterest: IDs of agents interested (1/0, -1 padding).
        isModeled: Whether agent's trajectory needs to be predicted for WOSAC (1/0, -1 padding).
        difficulty: Difficulty of the agent's trajectory to be predicted (0/1/2 if isModeled, 0 if !isModeled, -1 padding).
    """

    def __init__(self, metadata_tensor: torch.Tensor):
        """Initializes the Metadata with the metadata tensor."""
        self.is_sdc = metadata_tensor[:, :, 0]
        self.objects_of_interest = metadata_tensor[:, :, 1]
        self.tracks_to_predict = metadata_tensor[:, :, 2]
        self.difficulty = metadata_tensor[:, :, 3]

    @classmethod
    def from_tensor(
        cls, metadata_tensor: madrona_gpudrive.madrona.Tensor, backend="torch"
    ):
        """Creates a Metadata object from the metadata_tensor."""
        if backend == "torch":
            return cls(metadata_tensor.to_torch().clone())
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (num_worlds, num_agents) of each metadata mask."""
        return self.is_sdc.shape