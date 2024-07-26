import torch


class HumanExpertActor:
    def __init__(self, is_controlled_func):
        """Expert actor that uses the logged human trajectories."""
        self.is_controlled_func = is_controlled_func
        self.actor_ids = torch.where(is_controlled_func)[0]

    def select_action(self, obs):
        return torch.full((len(self.actor_ids), 1), torch.nan)
