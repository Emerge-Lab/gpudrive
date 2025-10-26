import copy
from typing import List, Union
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
import pufferlib.models #主要作用为正交初始化神经网络层
from gpudrive.env import constants
from huggingface_hub import PyTorchModelHubMixin
from box import Box

import madrona_gpudrive

TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount

#计算log概率
def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

#计算熵
def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

#给定 logits（动作概率），返回采样/选择的 action、对应的 logprob 与 entropy
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
        fusion_type="attention",  # 新增：融合类型选择
        num_attention_heads=4,  # 新增：注意力头数
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
        self.fusion_type = fusion_type
        self.num_attention_heads = num_attention_heads

        # Indices for unpacking the observation
        self.ego_state_idx = constants.EGO_FEAT_DIM
        self.partner_obs_idx = (
            constants.PARTNER_FEAT_DIM * self.max_controlled_agents
        )
        
        # Set default value for vbd_in_obs
        self.vbd_in_obs = False
        
        if config is not None:
            self.config = Box(config)
            if "reward_type" in self.config:
                if self.config.reward_type == "reward_conditioned":
                    # Agents know their "type", consisting of three weights
                    # that determine the reward (collision, goal, off-road)
                    self.ego_state_idx += 3
                    self.partner_obs_idx += 3

            # Override default if config contains vbd_in_obs
            if hasattr(self.config, 'vbd_in_obs'):
                self.vbd_in_obs = self.config.vbd_in_obs

        # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
        self.vbd_size = 91 * 5

        self.ego_embed = nn.Sequential(
            pufferlib.pytorch.layer_init( #初始化线性层
                nn.Linear(self.ego_state_idx, input_dim)
            ),
            nn.LayerNorm(input_dim), #层归一化
            self.act_func, #激活函数
            nn.Dropout(self.dropout), #丢弃，防止过拟合
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)), #初始化线性层
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

        # 新增：注意力融合机制
        if self.fusion_type == "attention":
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=self.num_attention_heads,
                dropout=self.dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(input_dim)
            # 注意力融合后的输出维度（使用flatten保留完整信息）
            fusion_output_dim = input_dim * 3
        elif self.fusion_type == "adaptive":
            # 自适应权重融合
            self.adaptive_weights = nn.Sequential(
                nn.Linear(input_dim * self.num_modes, 64),
                self.act_func,
                nn.Linear(64, self.num_modes),
                nn.Softmax(dim=-1)
            )
            fusion_output_dim = input_dim
        else:  # 原始简单拼接
            fusion_output_dim = self.input_dim * self.num_modes

        self.shared_embed = nn.Sequential(
            nn.Linear(fusion_output_dim, self.hidden_dim),
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

        # 新增：不同的融合策略
        if self.fusion_type == "attention":
            # 注意力融合
            embed = self._attention_fusion(ego_embed, partner_embed, road_map_embed)
        elif self.fusion_type == "adaptive":
            # 自适应权重融合
            embed = self._adaptive_fusion(ego_embed, partner_embed, road_map_embed)
        else:
            # 原始简单拼接
            embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.shared_embed(embed)

    def _attention_fusion(self, ego_embed, partner_embed, road_embed):
        """使用多头注意力机制进行模态融合"""
        # 组合所有模态: (batch, 3, input_dim)
        modalities = torch.stack([ego_embed, partner_embed, road_embed], dim=1)
        
        # 自注意力融合
        attended, attention_weights = self.attention_fusion(
            modalities, modalities, modalities
        )
        
        # 残差连接 + 层归一化
        attended = self.attention_norm(attended + modalities)
        
        # 使用flatten保留完整信息，而不是平均池化
        # 这样可以避免信息瓶颈（192维 vs 64维），提高最终性能
        return attended.flatten(start_dim=1) 
    
    def _adaptive_fusion(self, ego_embed, partner_embed, road_embed):
        """使用自适应权重进行模态融合"""
        # 拼接所有模态特征
        combined = torch.cat([ego_embed, partner_embed, road_embed], dim=-1)
        
        # 计算每个模态的权重
        weights = self.adaptive_weights(combined)
        
        # 加权融合
        modalities = torch.stack([ego_embed, partner_embed, road_embed], dim=-1)
        weighted_fusion = (modalities * weights.unsqueeze(1)).sum(dim=-1)
        
        return weighted_fusion

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
