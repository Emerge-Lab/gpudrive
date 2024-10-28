import logging
import time
import wandb
import torch
from torch.nn import functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

# Import masked rollout buffer class
from algorithms.sb3.rollout_buffer import MaskedRolloutBuffer
from networks.perm_eq_late_fusion import LateFusionNet

# From stable baselines
def explained_variance(
    y_pred: torch.tensor, y_true: torch.tensor
) -> torch.tensor:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = torch.var(y_true)
    return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y


class IPPO(PPO):
    """Adapted Proximal Policy Optimization algorithm (PPO) that is compatible with multi-agent environments."""

    def __init__(
        self,
        *args,
        env_config=None,
        exp_config=None,
        mlp_class: nn.Module = LateFusionNet,
        mlp_config=None,
        **kwargs,
    ):
        self.env_config = env_config
        self.exp_config = exp_config
        self.mlp_class = mlp_class
        self.mlp_config = mlp_config
        self.resample_counter = 0
        super().__init__(*args, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskedRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Adapted collect_rollouts function."""

        assert (
            self._last_obs is not None
        ), "No previous observation was provided"

        # Check resampling criterion and resample batch of scenarios if needed
        if self.env.exp_config.resample_scenarios:
            if self.env.exp_config.resample_criterion == "global_step":
                if self.resample_counter >= self.env.exp_config.resample_freq:
                    print(
                        f"Resampling {self.env.num_worlds} scenarios at global_step {self.num_timesteps:,}..."
                    )
                    # Re-initialize the scenes and controlled agents mask
                    self.env.resample_scenario_batch()
                    self.resample_counter = 0
                    # Get new initial observation
                    self._last_obs = self.env.reset()
                    # Update storage shapes
                    self.n_envs = env.num_valid_controlled_agents_across_worlds
                    rollout_buffer.n_envs = self.n_envs
                    self._last_episode_starts = (
                        self.env._env.get_dones().clone()[
                            ~self.env.dead_agent_mask
                        ]
                    )

            else:
                raise NotImplementedError(
                    f"Resampling criterion {self.env.exp_config.resample_criterion} not implemented"
                )

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        time_rollout = time.perf_counter()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = self._last_obs

                # EDIT_1: Mask out invalid observations (NaN axes and/or dead agents)
                # Create dummy actions, values and log_probs (NaN)
                actions = torch.full(
                    fill_value=float("nan"), size=(self.n_envs,)
                ).to(self.device)
                log_probs = torch.full(
                    fill_value=float("nan"),
                    size=(self.n_envs,),
                    dtype=torch.float32,
                ).to(self.device)
                values = (
                    torch.full(
                        fill_value=float("nan"),
                        size=(self.n_envs,),
                        dtype=torch.float32,
                    )
                    .unsqueeze(dim=1)
                    .to(self.device)
                )

                # Get indices of alive agent ids
                # Convert env_dead_agent_mask to boolean tensor with the same shape as obs_tensor
                alive_agent_mask = ~(
                    env.dead_agent_mask[env.controlled_agent_mask].reshape(
                        env.num_envs, 1
                    )
                )

                # Use boolean indexing to select elements in obs_tensor
                obs_tensor_alive = obs_tensor[
                    alive_agent_mask.expand_as(obs_tensor)
                ].reshape(-1, obs_tensor.shape[-1])

                # Predict actions, vals and log_probs given obs
                time_actions = time.perf_counter()
                actions_tmp, values_tmp, log_prob_tmp = self.policy(
                    obs_tensor_alive
                )
                nn_fps = actions_tmp.shape[0] / (
                    time.perf_counter() - time_actions
                )
                self.logger.record("rollout/nn_fps", nn_fps)

                # Predict actions, vals and log_probs given obs
                (
                    actions[alive_agent_mask.squeeze(dim=1)],
                    values[alive_agent_mask.squeeze(dim=1)],
                    log_probs[alive_agent_mask.squeeze(dim=1)],
                ) = (
                    actions_tmp.float(),
                    values_tmp.float(),
                    log_prob_tmp.float(),
                )

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(
                        clipped_actions
                    )
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = torch.clamp(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # EDIT_2: Increment the global step by the number of valid samples in rollout step
            self.num_timesteps += int((~rewards.isnan()).float().sum().item())
            self.resample_counter += int(
                (~rewards.isnan()).float().sum().item()
            )
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                torch.Tensor(self._last_episode_starts),  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # # # # # END LOOP # # # # #
        total_steps = self.n_envs * n_rollout_steps
        elapsed_time = time.perf_counter() - time_rollout
        fps = total_steps / elapsed_time
        self.logger.record("charts/fps", fps)

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(new_obs)  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Change buffer to our own masked version
        buffer_cls = MaskedRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        if self.mlp_class == LateFusionNet:
            self.policy = self.policy_class(
                observation_space=self.observation_space,
                env_config=self.env_config,
                exp_config=self.exp_config,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
                use_sde=self.use_sde,
                mlp_class=self.mlp_class,
                mlp_config=self.mlp_config,
                **self.policy_kwargs,
            )
        else:
            self.policy = self.policy_class(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
                use_sde=self.use_sde,
                mlp_class=self.mlp_class,
                **self.policy_kwargs,
            )

        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu()
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
        self.logger.record("train/explained_var", explained_var.item())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/advantages", advantages.mean().item())
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
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
