import torch
from pathlib import Path
from algorithms.sb3.ppo.ippo import IPPO


class SB3PolicyActor:
    """Policy actor that selects actions based on a learned policy."""

    def __init__(
        self,
        is_controlled_func,
        saved_model_path,
        deterministic=True,
        device="cuda",
    ):
        self.is_controlled_func = is_controlled_func
        self.actor_ids = torch.where(is_controlled_func)[0]
        self.device = device
        self.deterministic = deterministic
        self.policy = self.load_model(saved_model_path)

    def load_model(self, saved_model_path):
        """Load a learned policy."""
        model_file = Path(saved_model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"File not found: {saved_model_path}")
        else:
            policy = IPPO.load(
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
            obs[self.is_controlled_func, :],
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
