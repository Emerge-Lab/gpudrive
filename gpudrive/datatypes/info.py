import torch
import madrona_gpudrive


class Info:
    """A class to represent the information about the state of the environment.
    Initialized from info_tensor (src/bindings) of shape (num_worlds, max_agents_in_scene, 6).
    For details, see `Info` in src/types.hpp.
    
    Info tensor structure (matches src/types.hpp):
    - [:, :, 0]: collidedWithRoad
    - [:, :, 1]: collidedWithRoadEdge
    - [:, :, 2]: collidedWithVehicle
    - [:, :, 3]: collidedWithNonVehicle
    - [:, :, 4]: reachedGoal
    - [:, :, 5]: type
    """

    def __init__(self, info_tensor: torch.Tensor):
        """Initializes the ego state with an observation tensor."""
        # 从tensor中提取各个字段（匹配C++ Info结构体的顺序）
        self.off_road = info_tensor[:, :, 0:2].sum(axis=2)
        self.collided = info_tensor[:, :, 2:4].sum(axis=2)
        self.goal_achieved = info_tensor[:, :, 4]
        self.off_road_with_road_edge: torch.Tensor = info_tensor[:, :, 1].to(torch.float)
        self.type: torch.Tensor = info_tensor[:, :, 5]
        
    @classmethod
    def from_tensor(
        cls,
        info_tensor: madrona_gpudrive.madrona.Tensor,
        backend="torch",
        device="cuda",
    ):
        """Creates an LocalEgoState from the agent_observation_tensor."""
        if backend == "torch":
            return cls(info_tensor.to_torch().clone().to(device))
        elif backend == "jax":
            raise NotImplementedError("JAX backend not implemented yet.")

    @property
    def shape(self):
        """Returns the shape of the info tensor (num_worlds, max_agents_in_scene)."""
        return self.off_road.shape
