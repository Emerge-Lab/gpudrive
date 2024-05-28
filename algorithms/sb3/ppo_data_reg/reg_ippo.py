import logging

import numpy as np
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch.nn import functional as F
from gymnasium import spaces

from algorithms.sb3.ppo.ippo import IPPO

logging.getLogger(__name__)


class RegularizedIPPO(IPPO):
    """Adapted Regularized PPO class compatible with multi-agent environments.
    Args:
        reg_policy (stable_baselines3.common.policies.ActorCriticPolicy): Regularization policy.
        reg_weight (float): Weight of regularization loss.
        reg_loss (torch.nn.Module): Regularization loss function.
    """

    def __init__(
        self,
        reg_policy=None,
        reg_weight=None,
        *,
        reg_loss=None,
        reg_weight_decay_schedule=None,  # linear, exponential
        reg_decay_rate=5,  # Used for exponential weight decay
        **kwargs,
    ):
        super().__init__(**kwargs)

        if reg_weight is None:
            logging.info(
                "No regularization weight specified, using default PPO."
            )
        elif not isinstance(reg_weight, float):
            raise TypeError(
                f"reg_weight must be float between 0.0 and 1.0, got {type(reg_weight)}."
            )
        elif not (0.0 <= reg_weight <= 1.0):
            raise ValueError(
                f"reg_weight must be float between 0.0 and 1.0, got {reg_weight}."
            )
        self.reg_weight = reg_weight
        self.reg_weight_decay_schedule = reg_weight_decay_schedule
        self.reg_decay_rate = reg_decay_rate

        if self.reg_weight is None and reg_policy is not None:
            logging.warning(
                "Regularization policy specified but no regularization weight, ignoring"
                " regularization policy."
            )
        self.reg_policy = reg_policy
        self.reg_loss = reg_loss

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # # # # # # # # # HR_PPO EDIT # # # # # # # # #
                if self.reg_weight is not None and self.reg_policy is not None:

                    # Get human policy action distributions conditioned on observations
                    reg_policy_action_dist = self.reg_policy.get_distribution(
                        rollout_data.observations
                    ).distribution.probs

                    # Get RL policy action distributions conditioned on observations
                    policy_action_dist = self.policy.get_distribution(
                        rollout_data.observations
                    ).distribution.probs

                    # Compute loss
                    loss_reg = self.reg_loss(
                        policy_action_dist.log(), reg_policy_action_dist.log()
                    )

                # # # # # # # # # HR_PPO EDIT # # # # # # # # #
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # # # # # # # # # HR_PPO EDIT # # # # # # # # #
                loss_ppo = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                if self.reg_weight is not None and self.reg_policy is not None:
                    # If weight decay schedule is defined, decay the weight over time
                    if self.reg_weight_decay_schedule is not None:
                        if self.reg_weight_decay_schedule == "linear":
                            reg_weight = (
                                self.reg_weight
                                * self._current_progress_remaining
                            )
                        elif self.reg_weight_decay_schedule == "exponential":
                            reg_weight = self.reg_weight * np.exp(
                                -self.reg_decay_rate
                                * (1 - self._current_progress_remaining)
                            )
                        elif self.reg_weight_decay_schedule == "None":
                            reg_weight = self.reg_weight
                    else:  # Always use the same regularization weight
                        reg_weight = self.reg_weight

                    # Define the loss as a weighted sum of the PPO loss and the regularization loss
                    loss = (1 - reg_weight) * loss_ppo + reg_weight * loss_reg
                else:
                    # Use default PPO loss
                    loss = loss_ppo

                # # # # # # # # # HR_PPO EDIT # # # # # # # # #

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                        .cpu()
                        .numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logs
        if self.reg_weight is not None and self.reg_policy is not None:
            self.logger.record("regularize/reg_weight", reg_weight)
            self.logger.record("regularize/loss_ppo", np.abs(loss_ppo.item()))
            self.logger.record("regularize/loss_kl", loss_reg.item())
            self.logger.record(
                "regularize/loss_kl_weighted", reg_weight * loss_reg.item()
            )
            self.logger.record(
                "regularize/loss_ppo_weighted",
                (1 - reg_weight) * np.abs(loss_ppo.item()),
            )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
