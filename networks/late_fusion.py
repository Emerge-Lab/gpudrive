from typing import List, Union
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
import pufferlib.models
from pygpudrive.env import constants

import gpudrive

def unpack_obs(obs_flat):
    """
    Unpack the flattened observation into the ego state, visible simulator state.

    Args:
        obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

    Returns:
        ego_state, road_objects, road_graph (torch.Tensor).
    """
    top_k_road_points = gpudrive.kMaxAgentMapObservationsCount

    ego_state = obs_flat[:, : constants.EGO_FEAT_DIM]
    vis_state = obs_flat[:, constants.EGO_FEAT_DIM :]

    ro_end_idx = constants.PARTNER_FEAT_DIM * constants.ROAD_GRAPH_FEAT_DIM
    rg_end_idx = ro_end_idx + (
        constants.ROAD_GRAPH_FEAT_DIM * top_k_road_points
    )

    road_objects = vis_state[:, :ro_end_idx].reshape(
        -1, constants.ROAD_GRAPH_FEAT_DIM, constants.PARTNER_FEAT_DIM
    )
    road_graph = vis_state[:, ro_end_idx:rg_end_idx].reshape(
        -1, top_k_road_points, constants.ROAD_GRAPH_FEAT_DIM
    )

    return ego_state, road_objects, road_graph


def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    action=None,
    deterministic=False,
):
    """Sample logits: Supports deterministic sampling."""
    
    normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
    logits = [logits]

    if action is None:
        if deterministic:
            # Select the action with the maximum probability
            action = torch.stack(
                [l.argmax(dim=-1) for l in logits]
            )
        else:
            # Sample actions stochastically from the logits
            action = torch.stack(
                [
                    torch.multinomial(logits_to_probs(l), 1).squeeze()
                    for l in logits
                ]
            )
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)

    logprob = torch.stack(
        [log_prob(l, a) for l, a in zip(normalized_logits, action)]
    ).T.sum(1)

    logits_entropy = torch.stack(
        [entropy(l) for l in normalized_logits]
    ).T.sum(1)

    return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=1, dropout=0.05):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.transformer_encoder(x))


class LateFusionTransformer(nn.Module):
    def __init__(
        self,
        action_dim,
        input_dim=64,
        hidden_dim=128,
        pred_heads_arch=[128],
        num_transformer_layers=0,
        dropout=0.00,
        act_func="tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.pred_heads_arch = pred_heads_arch
        self.num_transformer_layers = num_transformer_layers
        self.num_modes = 3  # Ego, partner, road graph
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()

        self.ego_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.EGO_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.partner_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.PARTNER_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.road_map_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.shared_embed = nn.Sequential(
            nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
            nn.Dropout(self.dropout)
        )

        if self.num_transformer_layers > 0:
            self.transformer_layers = nn.Sequential(
                *[
                    EncoderBlock(
                        hidden_dim, hidden_dim, nhead=1, dropout=dropout
                    )
                    for _ in range(self.num_transformer_layers)
                ]
            )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, action_dim), std=0.01
        )
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, 1), std=1
        )

    def encode_observations(self, observation):
        ego_state, road_objects, road_graph = unpack_obs(observation)

        ego_embed = self.ego_embed(ego_state)

        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        if self.num_transformer_layers > 0:
            embed = self.transformer_layers(embed)

        return self.shared_embed(embed)

    def forward(self, obs, action=None, deterministic=False):

        # Encode the observations
        hidden = self.encode_observations(obs)

        # Decode the actions
        value = self.critic(hidden)
        logits = self.actor(hidden)

        action, logprob, entropy = sample_logits(logits, action, deterministic)
        
        return action, logprob, entropy, value

    def _build_network(self, input_dim, net_arch, network_type):
        layers = []
        last_dim = input_dim
        for layer_dim in net_arch:
            layers.extend(
                [
                    nn.Linear(last_dim, layer_dim),
                    nn.Dropout(self.dropout),
                    nn.LayerNorm(layer_dim),
                    self.act_func,
                ]
            )
            last_dim = layer_dim

        output_dim = self.action_dim if network_type == "actor" else 1
        std = 0.01 if network_type == "actor" else 1.0
        layers.append(
            pufferlib.pytorch.layer_init(
                nn.Linear(last_dim, output_dim), std=std
            )
        )

        return nn.Sequential(*layers)