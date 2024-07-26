class SimAgentActor:
    def __init__(self, actor_ids):

        """Base class for GPUDrive simulation agents.

        Args:
            env: The environment.
            actor_id_tensor (torch.Tensor): Actor ids that the agent controls.
        """
        self.actor_ids = actor_ids

    def select_action(self, obs):
        raise NotImplementedError
