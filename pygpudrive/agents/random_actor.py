import torch


class RandomActor:
    """Random actor.

    Args:
        env: Environment.
        is_controlled_func (torch.Tensor): Determines which agents are controlled by this actor. Shape: (max_num_agents,).
        valid_agent_mask (torch.Tensor): Mask that determines which agents are valid, and thus controllable, in the environment. Shape: (num_worlds, max_num_agents).
        device (str): Device to put the actions on.
    """

    def __init__(
        self, env, is_controlled_func, valid_agent_mask, device="cuda"
    ):
        self.env = env
        self.is_controlled_func = is_controlled_func
        self.device = device
        self.valid_and_controlled_mask = self.get_valid_actor_mask(
            is_controlled_func, valid_agent_mask
        )
        self.actor_ids = [
            torch.where(self.valid_and_controlled_mask[world_idx, :])[0]
            for world_idx in range(valid_agent_mask.shape[0])
        ]

    def select_action(self):
        """Select random actions."""

        action_lists = []
        for world_idx in range(len(self.actor_ids)):

            actions = torch.Tensor(
                [
                    self.env.action_space.sample()
                    for _ in range(len(self.actor_ids[world_idx]))
                ]
            ).to(self.device)
            action_lists.append(actions)

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
