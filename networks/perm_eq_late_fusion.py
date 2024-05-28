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
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from pygpudrive.env.config import EnvConfig


os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "./gpudrive_cache"

class LateFusionNet(nn.Module):
    """Processes the env observation using a late fusion architecture."""

    def __init__(
        self,
        observation_space: spaces.Box,
        env_config: EnvConfig = None,
        obj_dims: List[int] = [6, 7, 7],
        arch_ego_state: List[int] = [10],
        arch_road_objects: List[int] = [64, 32],
        arch_road_graph: List[int] = [64, 32],
        arch_shared_net: List[int] = [256, 128, 64],
        act_func: str = "tanh",
        dropout: float = 0.0,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        # Unpack feature dimensions
        self.ego_input_dim = obj_dims[0]
        self.ro_input_dim = obj_dims[1]
        self.rg_input_dim = obj_dims[2]

        self.config = env_config
        self._set_obj_dims(env_config)

        # Network architectures
        self.arch_ego_state = arch_ego_state
        self.arch_road_objects = arch_road_objects
        self.arch_road_graph = arch_road_graph
        self.arch_shared_net = arch_shared_net
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()
        self.dropout = dropout

        # Save output dimensions, used to create the action distribution & value
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # If using max pool across object dim
        self.shared_net_input_dim = (
            arch_ego_state[-1] + arch_road_objects[-1] + arch_road_graph[-1]
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

    def _build_out_network(self, input_dim: int, output_dim: int, net_arch: List[int]):
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

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        road_objects = F.max_pool1d(road_objects.permute(0, 2, 1), kernel_size=self.ro_max).squeeze(-1)
        road_graph = F.max_pool1d(road_graph.permute(0, 2, 1), kernel_size=self.rg_max).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        out = self.actor_out_net(torch.cat((ego_state, road_objects, road_graph), dim=1))

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
        road_objects = F.max_pool1d(road_objects.permute(0, 2, 1), kernel_size=self.ro_max).squeeze(-1)
        road_graph = F.max_pool1d(road_graph.permute(0, 2, 1), kernel_size=self.rg_max).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        out = self.val_out_net(torch.cat((ego_state, road_objects, road_graph), dim=1))

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

        # Visible state object order: road_objects, road_points, traffic_lights, stop_signs
        # Find the ends of each section
        ro_end_idx = self.ro_input_dim * self.ro_max
        rg_end_idx = ro_end_idx + (self.rg_input_dim * self.rg_max)

        # Unflatten and reshape to (batch_size, num_objects, object_dim)
        road_objects = (vis_state[:, :ro_end_idx]).reshape(-1, self.ro_max, self.ro_input_dim)
        road_graph = (vis_state[:, ro_end_idx:rg_end_idx]).reshape(
            -1,
            self.rg_max,
            self.rg_input_dim,
        )

        return ego_state, road_objects, road_graph

    def _set_obj_dims(self, config):
        # Define original object dimensions
        # TODO(ev) this doesn't exist in the config so remove hardcoding
        self.ro_max = 127 # config.partner_obs_dim
        self.rg_max = 2000 # config.map_obs_dim


class LateFusionPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        env_config: Box,
        mlp_class: Type[LateFusionNet] = LateFusionNet,
        mlp_config: Optional[Box] = None,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.observation_space = observation_space
        self.env_config = env_config
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
            **self.mlp_config,
        )


if __name__ == "__main__":
    # Import adapted PPO version
    from algorithms.sb3.ppo.ippo import IPPO
    from baselines.config import ExperimentConfig
    # Import the EnvConfig dataclass
    from pygpudrive.env.config import EnvConfig
    # Load configs
    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        sample_method="pad_n",
    )

    exp_config = ExperimentConfig(
        render=True,
    )

    # Make SB3-compatible environment
    env = SB3MultiAgentEnv(
        config=env_config,
        num_worlds=3,
        max_cont_agents=2,
        data_dir="waymo_data",
        device=exp_config.device,
    )
    obs = env.reset()
    obs = torch.Tensor(obs)[:2]

    model_config = None

    # Test
    run_id = None

    model = IPPO(
        env_config=env_config,
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
        device=exp_config.device,
        tensorboard_log=f"runs/{run_id}"
        if run_id is not None
        else None,  # Sync with wandb
        mlp_class=LateFusionNet,
        mlp_config=model_config,
        policy=LateFusionPolicy
    )

    model.learn(5000)
    env.close()
