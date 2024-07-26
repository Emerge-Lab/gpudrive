import torch


class RandomActor:
    """Random actor."""

    def __init__(self, env, is_controlled_func, device="cuda"):
        self.env = env
        self.is_controlled_func = is_controlled_func
        self.actor_ids = torch.where(is_controlled_func)[0]
        self.device = device

    def select_action(self):
        """Select random actions."""
        actions = torch.Tensor(
            [
                self.env.action_space.sample()
                for _ in range(len(self.actor_ids))
            ]
        ).to(self.device)

        return actions
