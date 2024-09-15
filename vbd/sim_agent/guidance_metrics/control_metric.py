import torch
from torch import nn


class ControlReward(nn.Module):
    def __init__(self, weight_a=1.0, weight_yaw=1.0) -> None:
        super().__init__()
        self.weight_a = weight_a
        self.weight_yaw = weight_yaw

    def forward(
        self, action_pred: torch.Tensor, c: dict, aoi: list = None, **kwargs
    ):
        """
        action_normalized: [B, A, T, 2]
        c: dict
        """
        mask = ~c["agents_mask"]  # [B, A]

        # B, A, T, 2 -> B, A, T
        cost = (
            action_pred[..., 0] ** 2 * self.weight_a
            + action_pred[..., 1] ** 2 * self.weight_yaw
        )

        cost = cost * mask[..., None]

        return -cost
