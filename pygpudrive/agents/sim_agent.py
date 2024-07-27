import torch
class SimAgentActor:
    """Base class for GPUDrive torch simulation agents.

    Args:
        env: The environment.
        is_controlled_func (torch.Tensor): A boolean tensor indicating which agents are controlled by the actors.
        device (str): The device.    
    """
    def __init__(self, env, is_controlled_func, device="cuda"):
        self.env = env
        self.is_controlled_func = is_controlled_func
        self.actor_ids = torch.where(is_controlled_func)[0]
        self.device = device
    
    def select_action(self, obs) -> torch.Tensor:
        """Select an action based on an observation.

        Args:
            obs (torch.Tensor): Batch of observations of shape (num_samples, observation_dim).

        Returns:
            torch.Tensor: _description_
        """
        raise NotImplementedError
