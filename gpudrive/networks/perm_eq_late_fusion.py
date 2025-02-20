import copy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import os

import torch
import torch.nn.functional as F
from box import Box
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import wandb

# Import env wrapper that makes gym env compatible with stable-baselines3
from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from gpudrive.env.config import EnvConfig
from gpudrive.env import constants


class LateFusionNet(nn.Module):
    """Processes the env observation using a late fusion architecture."""

    def __init__(
        self,
        observation_space: spaces.Box,
        env_config: EnvConfig,
        exp_config,
    ):
        super().__init__()

        self.config = env_config
        self.net_config = exp_config

        # Unpack feature dimensions
        self.ego_input_dim = constants.EGO_FEAT_DIM if self.config.ego_state else 0
        self.ro_input_dim = constants.PARTNER_FEAT_DIM if self.config.partner_obs else 0
        self.rg_input_dim = constants.ROAD_GRAPH_FEAT_DIM if self.config.road_map_obs else 0

        self.ro_max = self.config.max_num_agents_in_scene-1
        self.rg_max = self.config.roadgraph_top_k

        # Network architectures
        self.arch_ego_state = self.net_config.ego_state_layers
        self.arch_road_objects = self.net_config.road_object_layers
        self.arch_road_graph = self.net_config.road_graph_layers
        self.arch_shared_net = self.net_config.shared_layers
        self.act_func = (
            nn.Tanh() if self.net_config.act_func == "tanh" else nn.ReLU()
        )
        self.dropout = self.net_config.dropout

        # Save output dimensions, used to create the action distribution & value
        self.latent_dim_pi = self.net_config.last_layer_dim_pi
        self.latent_dim_vf = self.net_config.last_layer_dim_vf

        # If using max pool across object dim
        self.shared_net_input_dim = (
            self.net_config.ego_state_layers[-1]
            + self.net_config.road_object_layers[-1]
            + self.net_config.road_graph_layers[-1]
        )

        # Build the networks
        # Actor network
        self.actor_ego_state_net = self._build_network(
            input_dim=self.ego_input_dim,
            net_arch=self.arch_ego_state,
        )
        self.actor_ro_net = self._build_network(
            input_dim=self.ro_input_dim,
            net_arch=self.arch_road_objects,
        )
        self.actor_rg_net = self._build_network(
            input_dim=self.rg_input_dim,
            net_arch=self.arch_road_graph,
        )
        self.actor_out_net = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=self.latent_dim_pi,
            net_arch=self.arch_shared_net,
        )

        # Value network
        self.val_ego_state_net = copy.deepcopy(self.actor_ego_state_net)
        self.val_ro_net = copy.deepcopy(self.actor_ro_net)
        self.val_rg_net = copy.deepcopy(self.actor_rg_net)
        self.val_out_net = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=self.latent_dim_vf,
            net_arch=self.arch_shared_net,
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

    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ):
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        return nn.Sequential(*layers)

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

        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(features)

        # Embed features
        ego_state = self.actor_ego_state_net(ego_state)
        road_objects = self.actor_ro_net(road_objects)
        road_graph = self.actor_rg_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        out = self.actor_out_net(
            torch.cat((ego_state, road_objects, road_graph), dim=1)
        )

        return out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the value network."""

        ego_state, road_objects, road_graph = self._unpack_obs(features)

        # Embed features
        ego_state = self.val_ego_state_net(ego_state)
        road_objects = self.val_ro_net(road_objects)
        road_graph = self.val_rg_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        out = self.val_out_net(
            torch.cat((ego_state, road_objects, road_graph), dim=1)
        )

        return out

    def _unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
        """

        # Unpack ego and visible state
        ego_state = obs_flat[:, : self.ego_input_dim]
        vis_state = obs_flat[:, self.ego_input_dim :]

        # Visible state object order: road_objects, road_points
        # Find the ends of each section
        ro_end_idx = self.ro_input_dim * self.ro_max
        rg_end_idx = ro_end_idx + (self.rg_input_dim * self.rg_max)

        # Unflatten and reshape to (batch_size, num_objects, object_dim)
        road_objects = (vis_state[:, :ro_end_idx]).reshape(
            -1, self.ro_max, self.ro_input_dim
        )
        road_graph = (vis_state[:, ro_end_idx:rg_end_idx]).reshape(
            -1,
            self.rg_max,
            self.rg_input_dim,
        )

        return ego_state, road_objects, road_graph


class LateFusionPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        env_config: Box,
        exp_config: Box,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        mlp_class: Type[LateFusionNet] = LateFusionNet,
        mlp_config: Optional[Box] = None,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.observation_space = observation_space
        self.env_config = env_config
        self.exp_config = exp_config
        self.mlp_class = mlp_class
        self.mlp_config = mlp_config if mlp_config is not None else Box({})
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
        self.mlp_extractor = self.mlp_class(
            self.observation_space,
            self.env_config,
            self.exp_config,
            **self.mlp_config,
        )
