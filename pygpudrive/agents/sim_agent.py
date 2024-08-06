import torch


class SimAgentActor:
    """Base class for GPUDrive torch simulation agents.

    Args:
        is_controlled_func (torch.Tensor): Determines which agents are controlled by this actor (across worlds).
        valid_agent_mask (torch.Tensor): Mask that determines which agents are valid, and thus controllable, in the environment. Shape: (num_worlds, num_agents).
        device (str): The device.
    """

    def __init__(self, is_controlled_func, valid_agent_mask, device="cuda"):
        self.is_controlled_func = is_controlled_func
        self.device = device
        self.valid_and_controlled_mask = self.get_valid_actor_mask(
            is_controlled_func, valid_agent_mask
        )
        self.actor_ids = [
            torch.where(self.valid_and_controlled_mask[world_idx, :])[0]
            for world_idx in range(valid_agent_mask.shape[0])
        ]

    def select_action(self, obs) -> torch.Tensor:
        """Select an action based on an observation.

        Args:
            obs (torch.Tensor): Batch of observations of shape (num_samples, observation_dim).

        Returns:
            torch.Tensor: _description_
        """
        raise NotImplementedError

    def get_valid_actor_mask(self, is_controlled_func, valid_agent_mask):
        """Returns a boolean mask across worlds that indicates which agents
        are valid and controlled by this actor.
        """
        num_worlds = valid_agent_mask.shape[0]

        is_controlled_func = is_controlled_func.expand((num_worlds, -1))

        assert (
            is_controlled_func.shape == valid_agent_mask.shape
        ), f"is_controlled_func and valid_agent_mask must match but are not: {is_controlled_func.shape} vs {valid_agent_mask.shape}"

        return is_controlled_func.to(self.device) & valid_agent_mask.to(
            self.device
        )
