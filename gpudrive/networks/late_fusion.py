import copy
from typing import List, Union
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
import pufferlib.models
from gpudrive.env import constants
from huggingface_hub import PyTorchModelHubMixin
from box import Box

import madrona_gpudrive

TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount


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
            action = torch.stack([l.argmax(dim=-1) for l in logits])
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


class NeuralNet(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Emerge-Lab/gpudrive",
    docs_url="https://arxiv.org/abs/2502.14706",
    tags=["ffn"],
):
    def __init__(
        self,
        action_dim=91,  # Default: 7 * 13
        input_dim=64,
        hidden_dim=128,
        dropout=0.00,
        act_func="tanh",
        max_controlled_agents=64,
        obs_dim=None,  # Size of the flattened observation vector (calculated if None)
        config=None,  # Optional config
        oracle_mode=False,  # Enable oracle to observe co-player conditioning
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.num_modes = 3  # Ego, partner, road graph
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()
        self.oracle_mode = oracle_mode
        self.vbd_in_obs = config.get('vbd_in_obs', False) if config else False
        # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
        self.vbd_size = 91 * 5

        # Indices for unpacking the observation
        self.ego_state_idx = constants.EGO_FEAT_DIM
        self.partner_obs_idx = (
            constants.PARTNER_FEAT_DIM * self.max_controlled_agents
        )
        
        # Calculate extra dimensions needed for conditioning
        conditioning_dims = self._get_conditioning_dims(config)
        self.ego_state_idx += conditioning_dims
        self.partner_obs_idx += conditioning_dims
        
        # Calculate obs_dim if not provided
        if obs_dim is None:
            road_graph_size = madrona_gpudrive.kMaxAgentMapObservationsCount * constants.ROAD_GRAPH_FEAT_DIM
            self.obs_dim = self.partner_obs_idx + road_graph_size
            if self.vbd_in_obs:
                self.obs_dim += self.vbd_size
        else:
            self.obs_dim = obs_dim
        
        if config is not None:
            self.config = Box(config)

        if self.oracle_mode:
            has_reward = has_entropy = False
            if config:
                ctype = config.get("condition_type", "all")
                has_reward = ctype in ("reward", "all")
                has_entropy = ctype in ("entropy", "all")
            
            self.oracle_conditioning_size = (3 if has_reward else 0) + (1 if has_entropy else 0)
            assert self.oracle_conditioning_size > 0
            self.oracle_max_co_players = self.max_controlled_agents - 1
            self.oracle_total_conditioning_size = self.oracle_max_co_players * self.oracle_conditioning_size
            
            self.oracle_conditioning_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(self.oracle_conditioning_size, input_dim)),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

        self.ego_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.ego_state_idx, input_dim)
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

        if self.vbd_in_obs:
            self.vbd_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.vbd_size, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

        if self.oracle_mode:
            self.num_modes += 1

        shared_input_dim = self.input_dim * self.num_modes
        self.shared_embed = nn.Sequential(
            nn.Linear(shared_input_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, action_dim), std=0.01
        )
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, 1), std=1
        )

    def _get_conditioning_dims(self, config):
        """Calculate the number of extra features needed for conditioning."""
        extra_dims = 0
        if config is not None:
            config_box = Box(config)
            ctype = config_box.get('condition_type', 'all')
            if ctype in ('reward', 'all'):
                extra_dims += 3
            if ctype in ('entropy', 'all'):
                extra_dims += 1
        return extra_dims

    def encode_observations(self, observation):

        vbd_predictions = oracle_conditioning = None
        if self.vbd_in_obs and not self.oracle_mode:
            ego_state, road_objects, road_graph, vbd_predictions = self.unpack_obs(observation)
        elif not self.vbd_in_obs and self.oracle_mode:
            ego_state, road_objects, road_graph, oracle_conditioning = self.unpack_obs(observation)
        elif self.vbd_in_obs and self.oracle_mode:
            ego_state, road_objects, road_graph, vbd_predictions, oracle_conditioning = self.unpack_obs(observation)
        else:
            ego_state, road_objects, road_graph = self.unpack_obs(observation)

        # Embed the ego state
        ego_embed = self.ego_embed(ego_state)

        if self.vbd_in_obs:
            vbd_embed = self.vbd_embed(vbd_predictions)
            # Concatenate the VBD predictions with the ego state embedding
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        # Max pool
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        if self.oracle_mode:
            assert oracle_conditioning is not None
            conditioning_embed, _ = self.oracle_conditioning_embed(oracle_conditioning).max(dim=1)
            embed = torch.cat([embed, conditioning_embed], dim=1)

        return self.shared_embed(embed)

    def forward(self, obs, action=None, deterministic=False):
        hidden = self.encode_observations(obs)
        value = self.critic(hidden)
        logits = self.actor(hidden)

        action, logprob, entropy = sample_logits(logits, action, deterministic)

        return action, logprob, entropy, value

    def unpack_obs(self, obs_flat):
        """
            Unpack flattened observations into
              1) ego_state
              2) partner_obs
              3) road_graph
              4) vbd_predictions           (if `self.vbd_in_obs` is True)
              5) oracle_conditioning_3d    (if `self.oracle_mode` is True)
        """

        ego_state = obs_flat[:, :self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx:self.partner_obs_idx]
        roadgraph_start = self.partner_obs_idx
        roadgraph_end = ego_state.shape[1] - (self.vbd_size if self.vbd_in_obs else 0) - (self.oracle_total_conditioning_size if self.oracle_mode else 0)
        roadgraph_obs = obs_flat[:, roadgraph_start:roadgraph_end]

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
        )

        result = [ego_state, road_objects, road_graph]

        vbd_predictions = oracle_conditioning = None
        if self.vbd_in_obs and self.oracle_mode:
            vbd_predictions = obs_flat[:, -(self.vbd_size + self.oracle_total_conditioning_size):-self.oracle_total_conditioning_size]
            oracle_conditioning = obs_flat[:, -self.oracle_total_conditioning_size:]
        elif self.vbd_in_obs:
            vbd_predictions = obs_flat[:, -self.vbd_size:]
        elif self.oracle_mode:
            oracle_conditioning = obs_flat[:, -self.oracle_total_conditioning_size:]

        if self.vbd_in_obs:
            result.append(vbd_predictions)
        if self.oracle_mode:
            oracle_cond = oracle_conditioning.view(-1, self.oracle_max_co_players, self.oracle_conditioning_size)
            result.append(oracle_cond)
        return tuple(result)
