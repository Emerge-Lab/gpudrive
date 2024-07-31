import torch
from pathlib import Path
from algorithms.sb3.ppo.ippo import IPPO


class PolicyActor:
    """Policy actor that selects actions based on a learned policy.

    This class is compatible with policies trained with stable-baselines 3, such as PPO.

    Args:
        is_controlled_func (torch.Tensor): Determines which agents are controlled by this actor.
        valid_agent_mask (torch.Tensor): Mask that determines which agents are valid, and thus controllable, in the environment.
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
        self.is_valid_and_controlled_func = (
            is_controlled_func & valid_agent_mask.squeeze(dim=0)
        )
        self.actor_ids = torch.where(self.is_valid_and_controlled_func)[0]
        self.device = device
        self.deterministic = deterministic
        self.model_class = model_class
        self.policy = self.load_model(saved_model_path)

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
        obs = self._reshape_observation(obs)

        actions = self.policy._predict(
            obs[self.is_valid_and_controlled_func, :],
            deterministic=self.deterministic,
        )
        return actions

    def get_distribution(self, obs):
        """Get policy distribution for given observation."""
        return self.policy.get_distribution(obs)

    def evaluate_actions(self, obs, actions):
        """Evaluate actions."""
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        return values, log_prob, entropy

    def _reshape_observation(self, obs):
        """Verify observation shape"""

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs
        elif obs.dim() == 3:
            # Flatten over env x agents
            obs = obs.view(-1, obs.size(-1))
        else:
            raise ValueError(
                f"Expected obs to have 2 dimensions (num_envs x agents, obs_dim), but got {obs.dim()}."
            )

        return obs
