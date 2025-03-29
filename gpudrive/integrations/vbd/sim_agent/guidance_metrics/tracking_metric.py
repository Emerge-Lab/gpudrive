import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss


class TrackingReward(nn.Module):
    def __init__(self, loss_fn=smooth_l1_loss):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        traj_pred: torch.Tensor,
        traj_ref: torch.Tensor,
        weight: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of the metrics module.

        Args:
            traj_pred (torch.Tensor): The input tensor.
            traj_ref (torch.Tensor): The traj_reference tensor.
            weight (torch.Tensor): The weight tensor.

        Returns:
            torch.Tensor: The computed loss tensor.

        Raises:
            AssertionError: If traj_pred and traj_ref do not have the same shape.
            ValueError: If the weight shape is not compatible with traj_ref.

        """
        if weight is None:
            weight = torch.ones_like(traj_ref)

        assert (
            traj_pred.shape[:-1] == traj_ref.shape[:-1]
        ), f"traj_pred {traj_pred.shape} and traj_ref {traj_ref.shape} must have the same shape"
        d = traj_ref.shape[-1]

        if len(weight.shape) == (len(traj_ref.shape) - 1):
            weight = weight.unsqueeze(-1)
        elif len(weight.shape) == len(traj_ref.shape):
            assert (
                weight.shape[-1] == traj_ref.shape[-1]
            ), "weight shape must be either (batch, seq) or same as traj_ref"
        else:
            raise ValueError(
                "weight shape must be either (B, A, T) or same as traj_ref"
            )

        rewards = (
            -self.loss_fn(
                input=traj_pred[..., :d],
                target=traj_ref[..., :d],
                reduction="none",
            )
            * weight
        )

        return rewards


class GoalReward(nn.Module):
    def __init__(self, loss_fn=smooth_l1_loss):
        self.loss_fn = loss_fn

        super().__init__()

    def forward(
        self,
        traj_pred: torch.Tensor,
        goal: torch.Tensor,
        goal_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of the metrics module.

        Args:
            traj_pred (torch.Tensor): The input tensor.
            traj_ref (torch.Tensor): The traj_reference tensor.
            weight (torch.Tensor): The weight tensor.

        Returns:
            torch.Tensor: The computed loss tensor.

        Raises:
            AssertionError: If traj_pred and traj_ref do not have the same shape.
            ValueError: If the weight shape is not compatible with traj_ref.

        """
        if goal_mask is None:
            goal_mask = torch.ones_like(goal)

        d = goal.shape[-1]
        look_ahead = kwargs.get("look_ahead", -1)
        rewards = (
            -self.loss_fn(
                input=traj_pred[..., look_ahead, :d],
                target=goal,
                reduction="none",
            )
            * goal_mask
        )

        return rewards


class AnchorReward(nn.Module):  # Does not work well
    def __init__(self, loss_fn=smooth_l1_loss):
        self.loss_fn = loss_fn
        super().__init__()

    def forward(
        self,
        traj_pred: torch.Tensor,
        traj_ref: torch.Tensor,
        weight: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of the metrics module.

        Args:
            traj_pred (torch.Tensor): The input tensor. [B, A, T, D]
            traj_ref (torch.Tensor): The traj_reference tensor. [B, A, D]
            weight (torch.Tensor): The weight tensor. [B, A] or [B, A, D]

        Returns:
            torch.Tensor: The computed loss tensor.

        Raises:
            AssertionError: If traj_pred and traj_ref do not have the same shape.
            ValueError: If the weight shape is not compatible with traj_ref.

        """
        if weight is None:
            weight = torch.ones_like(traj_ref)

        d = traj_ref.shape[-1]
        if len(weight.shape) == (len(traj_ref.shape) - 1):
            weight = weight.unsqueeze(-1)
        elif len(weight.shape) == len(traj_ref.shape):
            assert (
                weight.shape[-1] == traj_ref.shape[-1]
            ), "weight shape must be either (batch, seq) or same as traj_ref"
        else:
            raise ValueError(
                "weight shape must be either (B, A, T) or same as traj_ref"
            )

        traj_ref = traj_ref.unsqueeze(-2).repeat(1, 1, traj_pred.shape[-2], 1)
        weight = weight.unsqueeze(-2).repeat(1, 1, traj_pred.shape[-2], 1)

        rewards = (
            -self.loss_fn(
                input=traj_pred[..., :d],
                target=traj_ref[..., :d],
                reduction="none",
            )
            * weight
        )

        rewards, _ = torch.min(torch.sum(rewards, dim=-1), dim=-1)

        return rewards
