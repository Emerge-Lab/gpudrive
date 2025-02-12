import copy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class FFN(nn.Module):
    """Custom feedforward neural network."""

    def __init__(
        self,
        feature_dim: int,
        layers: List[int] = [128],
        act_func: str = "tanh",
        dropout: float = 0.0,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()

        # DON'T CHANGE: Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Actor network
        self.actor_net = self._build_network(
            input_dim=feature_dim,
            net_arch=layers + [last_layer_dim_pi],
        )

        # Value network
        self.critic_net = self._build_network(
            input_dim=feature_dim,
            net_arch=layers + [last_layer_dim_vf],
        )

    def _build_network(
        self,
        input_dim: int,
        net_arch: List[int],
    ) -> nn.Module:
        """Build a network with the specified architecture."""
        layers = []
        last_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(last_dim, layer_dim))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            last_dim = layer_dim
        return nn.Sequential(*layers)

    def train(self, mode):
        """Turn on updates to mean and standard deviation."""
        self.track_running_states = True

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): input tensor of shape (batch_size, feature_dim)
        Return:
            (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the actor network."""
        return self.actor_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the value network."""
        return self.critic_net(features)

    def update_running_mean_std(self, features: torch.Tensor) -> None:
        """Update the mean and standard deviation."""
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0)


class FeedForwardPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        mlp_class: Type[FFN] = FFN,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.mlp_class = mlp_class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # Build the network architecture
        self.mlp_extractor = self.mlp_class(self.features_dim)
