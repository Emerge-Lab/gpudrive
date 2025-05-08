import torch
import torch.nn as nn

import numpy as np

from torch.distributions.categorical import Categorical

import madrona_gpudrive
from gpudrive.env import constants

TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Actor-Critic agent with shared embedding networks and separate actor and critic heads.
    Expects a single observation vector as input.
    """

    def __init__(
        self,
        config,
        embed_dim,
        action_dim,
        act_func="tanh",
        dropout=0.00,
        top_k=3,  # Max pooling features to keep
    ):
        super().__init__()

        self.config = config
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()
        self.dropout = dropout
        self.action_dim = action_dim
        self.top_k = top_k

        # Indices for unpacking the different observation modalities
        self.ego_state_idx = constants.EGO_FEAT_DIM
        if self.config["reward_type"] == "reward_conditioned":
            self.ego_state_idx += 3
        if self.config["add_previous_action"]:
            self.ego_state_idx += 2

        self.max_controlled_agents = madrona_gpudrive.kMaxAgentCount
        self.max_observable_agents = self.max_controlled_agents - 1
        self.partner_obs_idx = self.ego_state_idx + (
            constants.PARTNER_FEAT_DIM * self.max_observable_agents
        )

        self.road_map_idx = self.partner_obs_idx + (
            constants.ROAD_GRAPH_TOP_K * constants.ROAD_GRAPH_FEAT_DIM
        )

        if self.config["guidance"]:
            self.guidance_feature_dim = 0
            # One-hot encoding signalling the next time step
            if self.config["add_reference_pos_xy"]:
                self.guidance_feature_dim += (
                    constants.LOG_TRAJECTORY_LENGTH * 2
                )
            if self.config["add_reference_speed"]:
                # TODO: Change to reference path-length (90) if using the vbd_amortized trajs
                self.guidance_feature_dim += constants.LOG_TRAJECTORY_LENGTH
            if self.config["add_reference_heading"]:
                self.guidance_feature_dim += constants.LOG_TRAJECTORY_LENGTH
            self.guidance_idx = self.road_map_idx + self.guidance_feature_dim

        # Shared embedding networks for both actor and critic
        self.ego_embed = nn.Sequential(
            layer_init(nn.Linear(self.ego_state_idx, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.partner_embed = nn.Sequential(
            layer_init(nn.Linear(constants.PARTNER_FEAT_DIM, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.road_map_embed = nn.Sequential(
            layer_init(nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.guidance_embed = nn.Sequential(
            layer_init(nn.Linear(self.guidance_feature_dim, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear((2 * top_k + 2) * embed_dim, 128)),
            nn.LayerNorm(128),
            self.act_func,
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64),
            self.act_func,
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear((2 * top_k + 2) * embed_dim, 128)),
            nn.LayerNorm(128),
            self.act_func,
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64),
            self.act_func,
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def forward(self, x, action=None):
        """Forward pass through the network.
        Args:
            x: Flattened observation vector of size [B, F].
            action (Optional): Actions to take of size [B, 1].
                If None, a new actions are sampled.
        """
        # Unpack into modalities
        ego_state, partner_obs, road_graph, guidance = self.unpack_obs(x)

        # Use shared embedding networks for both actor and critic
        ego_embed = self.ego_embed(ego_state)
        partner_embed = self.partner_embed(partner_obs)
        road_embed = self.road_map_embed(road_graph)
        guidance_embed = self.guidance_embed(guidance)

        # Take top k features from partner and road embeddings
        partner_max_pool = torch.topk(partner_embed, k=self.top_k, dim=1)[
            0
        ].flatten(
            start_dim=1
        )  # partner_embed.max(dim=1)
        road_max_pool = torch.topk(road_embed, k=self.top_k, dim=1)[0].flatten(
            start_dim=1
        )

        # Concatenate the embeddings
        z = torch.cat(
            [ego_embed, partner_max_pool, road_max_pool, guidance_embed],
            dim=-1,
        )

        # Pass to the actor and critic networks
        logits = self.actor(z)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        else:
            # Reshape action from [B, 1] to [B] to match what Categorical expects
            action = action.squeeze(-1)

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(z),
        )

    def unpack_obs(self, obs_flat):
        """Unpack the flattened observation into the ego state, visible simulator state."""

        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]
        roadgraph_obs = obs_flat[:, self.partner_obs_idx : self.road_map_idx]
        guidance = obs_flat[:, self.road_map_idx :]

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
        )

        return ego_state, road_objects, road_graph, guidance


class SeparateActorCriticAgent(nn.Module):
    """ "
    Actor-Critic agent with separate actor and critic networks.
    Expects a single observation vector as input.
    """

    def __init__(
        self,
        config,
        embed_dim,
        action_dim,
        act_func="tanh",
        dropout=0.00,
    ):
        super().__init__()

        self.config = config
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()
        self.dropout = dropout
        self.action_dim = action_dim

        # Indices for unpacking the observation modalities
        self.ego_state_idx = (
            9
            if self.config["reward_type"] == "reward_conditioned"
            else constants.EGO_FEAT_DIM
        )
        if self.config[
            "add_reference_pos_xy"
        ]:  # Every agent receives a reference path
            self.ego_state_idx += 91 * 2

        self.max_controlled_agents = madrona_gpudrive.kMaxAgentCount
        self.max_observable_agents = self.max_controlled_agents - 1
        self.partner_obs_idx = self.ego_state_idx + (
            constants.PARTNER_FEAT_DIM * self.max_observable_agents
        )

        # Actor's embedding networks
        self.actor_ego_embed = nn.Sequential(
            layer_init(nn.Linear(self.ego_state_idx, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.actor_partner_embed = nn.Sequential(
            layer_init(nn.Linear(constants.PARTNER_FEAT_DIM, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.actor_road_map_embed = nn.Sequential(
            layer_init(nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        # Critic's embedding networks
        self.critic_ego_embed = nn.Sequential(
            layer_init(nn.Linear(self.ego_state_idx, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.critic_partner_embed = nn.Sequential(
            layer_init(nn.Linear(constants.PARTNER_FEAT_DIM, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        self.critic_road_map_embed = nn.Sequential(
            layer_init(nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, embed_dim)),
            nn.LayerNorm(embed_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(3 * embed_dim, 32)),
            nn.LayerNorm(32),
            self.act_func,
            layer_init(
                nn.Linear(32, 1), std=1.0
            ),  # Fixed the dimension (was 32)
        )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(3 * embed_dim, 64)),
            nn.LayerNorm(64),
            self.act_func,
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        # Unpack into modalities
        ego_state, partner_obs, road_graph = self.unpack_obs(x)

        # Embed each modality using critic's embedding networks
        critic_ego_embed = self.critic_ego_embed(ego_state)
        critic_partner_embed = self.critic_partner_embed(partner_obs)
        critic_road_embed = self.critic_road_map_embed(road_graph)

        # Concatenate the embeddings
        critic_x = torch.cat(
            [critic_ego_embed, critic_partner_embed, critic_road_embed], dim=-1
        )

        return self.critic(critic_x)

    def forward(self, x, action=None):
        """Forward pass through the network.
        Args:
            x: Flattened observation vector of size [B, F].
            action (Optional): Actions to take of size [B, 1].
                If None, a new actions are sampled.
        """
        # Unpack into modalities
        ego_state, partner_obs, road_graph = self.unpack_obs(x)

        actor_ego_embed = self.actor_ego_embed(ego_state)
        actor_partner_embed, _ = self.actor_partner_embed(partner_obs).max(
            dim=1
        )
        actor_road_embed, _ = self.actor_road_map_embed(road_graph).max(dim=1)
        actor_z = torch.cat(
            [actor_ego_embed, actor_partner_embed, actor_road_embed], dim=-1
        )

        critic_ego_embed = self.critic_ego_embed(ego_state)
        critic_partner_embed, _ = self.critic_partner_embed(partner_obs).max(
            dim=1
        )
        critic_road_embed, _ = self.critic_road_map_embed(road_graph).max(
            dim=1
        )
        critic_z = torch.cat(
            [critic_ego_embed, critic_partner_embed, critic_road_embed], dim=-1
        )

        # Pass to the actor and critic networks
        logits = self.actor(actor_z)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        else:
            # Reshape action from [B, 1] to [B] to match what Categorical expects
            action = action.squeeze(-1)

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(critic_z),
        )

    def unpack_obs(self, obs_flat):
        """Unpack the flattened observation into the ego state, visible simulator state."""

        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]
        roadgraph_obs = obs_flat[:, self.partner_obs_idx :]

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
        )

        return ego_state, road_objects, road_graph
