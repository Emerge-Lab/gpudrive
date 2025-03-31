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
        obs_dim=2984,  # Size of the flattened observation vector (hardcoded)
        config=None,  # Optional config
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.obs_dim = obs_dim
        self.num_modes = 3  # Ego, partner, road graph
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()

        # Indices for unpacking the observation
        self.ego_state_idx = constants.EGO_FEAT_DIM
        self.partner_obs_idx = (
            constants.PARTNER_FEAT_DIM * self.max_controlled_agents
        )
        if config is not None:
            self.config = Box(config)
            if "reward_type" in self.config:
                if self.config.reward_type == "reward_conditioned":
                    # Agents know their "type", consisting of three weights
                    # that determine the reward (collision, goal, off-road)
                    self.ego_state_idx += 3
                    self.partner_obs_idx += 3

            self.vbd_in_obs = self.config.vbd_in_obs

        # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
        self.vbd_size = 91 * 5

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

        self.shared_embed = nn.Sequential(
            nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, action_dim), std=0.01
        )
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, 1), std=1
        )

    def encode_observations(self, observation):

        if self.vbd_in_obs:
            (
                ego_state,
                road_objects,
                road_graph,
                vbd_predictions,
            ) = self.unpack_obs(observation)
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

        # Concatenate the embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.shared_embed(embed)

    def forward(self, obs, action=None, deterministic=False):

        # Encode the observations
        hidden = self.encode_observations(obs)

        # Decode the actions
        value = self.critic(hidden)
        logits = self.actor(hidden)

        action, logprob, entropy = sample_logits(logits, action, deterministic)

        return action, logprob, entropy, value

    def unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state, visible simulator state.

        Args:
            obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

        Returns:
            tuple: If vbd_in_obs is True, returns (ego_state, road_objects, road_graph, vbd_predictions).
                Otherwise, returns (ego_state, road_objects, road_graph).
        """

        # Unpack modalities
        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

        if self.vbd_in_obs:
            # Extract the VBD predictions (last 455 elements)
            vbd_predictions = obs_flat[:, -self.vbd_size :]

            # The rest (excluding ego_state and partner_obs) is the road graph
            roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.vbd_size]
        else:
            # Without VBD, all remaining elements are road graph observations
            roadgraph_obs = obs_flat[:, self.partner_obs_idx :]

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
        )

        if self.vbd_in_obs:
            return ego_state, road_objects, road_graph, vbd_predictions
        else:
            return ego_state, road_objects, road_graph
