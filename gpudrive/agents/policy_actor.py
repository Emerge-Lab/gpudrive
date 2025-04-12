import torch
from pathlib import Path
from integrations.rl.sb3.ppo import IPPO


class PolicyActor:
    """Policy actor that selects actions based on a learned policy.

    This class is compatible with policies trained with stable-baselines 3, such as PPO.

    Args:
        is_controlled_func (torch.Tensor): Determines which agents are controlled by this actor (across worlds).
        valid_agent_mask (torch.Tensor): Mask that determines which agents are valid, and thus controllable, in the environment. Shape: (num_worlds, num_agents).
        saved_model_path (str): Path to the saved model.
        model_class: Model class to use.
        deterministic (bool): Whether to use deterministic actions.
        device (str): Device to run the policy on.
    """

    def __init__(
        self,
        is_controlled_func,
        valid_agent_mask,
        saved_model_path,
        model_class=IPPO,
        deterministic=True,
        device="cuda",
    ):
        self.is_controlled_func = is_controlled_func
        self.device = device
        self.deterministic = deterministic
        self.model_class = model_class
        self.policy = self.load_model(saved_model_path)
        self.valid_and_controlled_mask = self.get_valid_actor_mask(
            is_controlled_func, valid_agent_mask
        )
        self.actor_ids = [
            torch.where(self.valid_and_controlled_mask[world_idx, :])[0]
            for world_idx in range(valid_agent_mask.shape[0])
        ]

    def load_model(self, saved_model_path):
        """Load a learned policy."""
        model_file = Path(saved_model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"File not found: {saved_model_path}")
        else:
            policy = self.model_class.load(
                path=saved_model_path,
                device=self.device,
            ).policy
        return policy

    def select_action(self, obs):
        """Use learned policy to select actions.

        obs (torch.Tensor): Observation tensor.
        """

        assert (
            obs.dim() == 3
        ), f"Expected obs to be of shape (num_worlds, max_agents, obs_dim), but got {obs.dim()}."

        action_lists = []
        for world_idx in range(len(self.actor_ids)):
            observations = obs[world_idx, self.actor_ids[world_idx], :]
            if (
                len(observations) == 0
            ):  # Append empty tensor if no agents in this world are controlled
                actions = torch.tensor([]).to(self.device)
            else:
                actions = self.policy._predict(
                    obs[world_idx, self.actor_ids[world_idx], :],
                    deterministic=self.deterministic,
                )
            action_lists.append(actions)

        return action_lists

    def get_distribution(self, obs):
        """Get policy distribution for given observation."""
        return self.policy.get_distribution(obs)

    def evaluate_actions(self, obs, actions):
        """Evaluate actions."""
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        return values, log_prob, entropy

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
