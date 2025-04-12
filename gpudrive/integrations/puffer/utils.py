from torch import nn
import torch
import torch.nn.functional as F
from pdb import set_trace as T
import pufferlib.models
import numpy as np
from pygpudrive.env import constants

def unpack_obs(obs_flat, env):
    """
    Unpack the flattened observation into the ego state and visible state.
    Args:
        obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
    Return:
        ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
    """
    top_k_road_points = env.env.config.roadgraph_top_k

    # Unpack ego and visible state
    ego_state = obs_flat[:, : constants.EGO_FEAT_DIM]
    vis_state = obs_flat[:, constants.EGO_FEAT_DIM :]

    # Visible state object order: road_objects, road_points
    # Find the ends of each section
    ro_end_idx = constants.PARTNER_FEAT_DIM * constants.ROAD_GRAPH_FEAT_DIM
    rg_end_idx = ro_end_idx + (
        constants.ROAD_GRAPH_FEAT_DIM * top_k_road_points
    )

    # Unflatten and reshape to (batch_size, num_objects, object_dim)
    road_objects = (vis_state[:, :ro_end_idx]).reshape(
        -1, constants.ROAD_GRAPH_FEAT_DIM, constants.PARTNER_FEAT_DIM
    )
    road_graph = (vis_state[:, ro_end_idx:rg_end_idx]).reshape(
        -1,
        top_k_road_points,
        constants.ROAD_GRAPH_FEAT_DIM,
    )
    return ego_state, road_objects, road_graph


class Policy(nn.Module):
    def __init__(
        self, env, input_size=64, hidden_size=128, act_func="tanh", **kwargs
    ):
        super().__init__()
        self.env = env
        self.act_func = (
            torch.nn.Tanh() if act_func == "tanh" else torch.nn.ReLU()
        )

        self.ego_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.EGO_FEAT_DIM, input_size)
            ),
            nn.LayerNorm(input_size),
            self.act_func,
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.PARTNER_FEAT_DIM, input_size)
            ),
            self.act_func,
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_map_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_size)
            ),
            nn.LayerNorm(input_size),
            self.act_func,
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embed = pufferlib.pytorch.layer_init(
            nn.Linear(3 * input_size, hidden_size)
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        ego_state, road_objects, road_graph = unpack_obs(
            observations, self.env
        )
        ego_embed = self.ego_embed(ego_state)

        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.shared_embed(embed), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value


class LiDARPolicy(nn.Module):
    def __init__(
        self, env, input_size=600, hidden_size=128, act_func="tanh", **kwargs
    ):
        super().__init__()
        self.env = env
        self.act_func = (
            torch.nn.Tanh() if act_func == "tanh" else torch.nn.ReLU()
        )

        self.embed = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(input_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            self.act_func,
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )

    def forward(self, observations):
        hidden = self.embed(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def decode_actions(self, flat_hidden, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
