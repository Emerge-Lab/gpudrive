import torch

EXPERT_ACTION_VALUE = -10_000  # Indicates an expert action


class HumanExpertActor:
    def __init__(self, is_controlled_func, valid_agent_mask, device="cuda"):
        """Expert actor that uses the logged human trajectories."""
        self.is_controlled_func = is_controlled_func
        self.device = device
        self.valid_and_controlled_mask = self.get_valid_actor_mask(
            is_controlled_func, valid_agent_mask
        )
        self.actor_ids = [
            torch.where(self.valid_and_controlled_mask[world_idx, :])[0]
            for world_idx in range(valid_agent_mask.shape[0])
        ]

    def select_action(self, obs):
        action_lists = [
            torch.full((len(actor_ids),), EXPERT_ACTION_VALUE).to(self.device)
            for actor_ids in self.actor_ids
        ]
        return action_lists

    def get_valid_actor_mask(self, is_controlled_func, valid_agent_mask):
        """Returns a boolean mask across worlds that indicates which agents
        are valid _and_ controlled by this actor.
        """
        num_worlds = valid_agent_mask.shape[0]
        is_controlled_func = is_controlled_func.expand((num_worlds, -1))

        return is_controlled_func.to(self.device) & valid_agent_mask.to(
            self.device
        )
