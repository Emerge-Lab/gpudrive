import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from .model_utils import (
    batch_transform_trajs_to_local_frame,
    batch_transform_polylines_to_local_frame,
    batch_transform_trajs_to_global_frame,
    roll_out,
)


class Encoder(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.agent_encoder = AgentEncoder()
        self.map_encoder = MapEncoder()
        self.traffic_light_encoder = TrafficLightEncoder()
        self.relation_encoder = FourierEmbedding(input_dim=3)
        self.transformer_encoder = TransformerEncoder(layers=layers)

    def forward(self, inputs):
        # agent encoding
        agents = inputs["agents_history"]
        agents_type = inputs["agents_type"]
        agents_interested = inputs["agents_interested"]
        agents_local = batch_transform_trajs_to_local_frame(agents)

        encoded_agents = torch.stack(
            [
                self.agent_encoder(agents_local[:, i], agents_type[:, i])
                for i in range(agents.shape[1])
            ],
            dim=1,
        )
        agents_mask = torch.eq(agents_interested, 0)

        # map and traffic light encoding
        map_polylines = inputs["polylines"]
        map_polylines_local = batch_transform_polylines_to_local_frame(
            map_polylines
        )
        encoded_map_lanes = self.map_encoder(map_polylines_local)
        maps_mask = inputs["polylines_valid"].logical_not()

        traffic_lights = inputs["traffic_light_points"]
        encoded_traffic_lights = self.traffic_light_encoder(traffic_lights)
        traffic_lights_mask = torch.eq(traffic_lights.sum(-1), 0)

        # relation encoding
        relations = inputs["relations"]
        relations = self.relation_encoder(relations)

        # transformer encoding
        encoder_outputs = {}
        encoder_outputs["agents"] = agents
        encoder_outputs["anchors"] = inputs["anchors"]
        encoder_outputs["agents_type"] = agents_type
        encoder_outputs["agents_mask"] = agents_mask
        encoder_outputs["maps_mask"] = maps_mask
        encoder_outputs["traffic_lights_mask"] = traffic_lights_mask
        encoder_outputs["relation_encodings"] = relations

        encodings = self.transformer_encoder(
            relations,
            encoded_agents,
            encoded_map_lanes,
            encoded_traffic_lights,
            agents_mask,
            maps_mask,
            traffic_lights_mask,
        )
        encoder_outputs["encodings"] = encodings

        return encoder_outputs


class GoalPredictor(nn.Module):
    def __init__(self, future_len=80, action_len=5, agents_len=32):
        super().__init__()
        self._agents_len = agents_len
        self._future_len = future_len
        self._action_len = action_len

        self.attention_layers = nn.ModuleList(
            [CrossTransformer() for _ in range(4)]
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.act_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, (self._future_len // self._action_len) * 2),
        )
        self.score_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, inputs):
        anchors_points = inputs["anchors"][:, : self._agents_len]
        anchors = self.anchor_encoder(anchors_points)
        encodings = inputs["encodings"]
        query = encodings[:, : self._agents_len, None] + anchors

        num_batch, num_agents, num_queries, _ = query.shape

        mask = torch.cat(
            [
                inputs["agents_mask"],
                inputs["maps_mask"],
                inputs["traffic_lights_mask"],
            ],
            dim=-1,
        )
        relations = inputs["relation_encodings"]

        actions = []
        scores = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[0](
                query[:, i], encodings, relations[:, i], key_mask=mask
            )
            query_content = self.attention_layers[1](
                query_content, encodings, relations[:, i], key_mask=mask
            )
            query_content = query_content + query[:, i]
            query_content = self.attention_layers[2](
                query_content, encodings, relations[:, i], key_mask=mask
            )
            query_content = self.attention_layers[3](
                query_content, encodings, relations[:, i], key_mask=mask
            )
            actions.append(
                self.act_decoder(query_content).reshape(
                    num_batch,
                    num_queries,
                    self._future_len // self._action_len,
                    2,
                )
            )
            scores.append(self.score_decoder(query_content).squeeze(-1))

        actions = torch.stack(actions, dim=1)
        scores = torch.stack(scores, dim=1)

        return actions, scores

    def reset_agent_length(self, agents_len):
        self._agents_len = agents_len


class Denoiser(nn.Module):
    def __init__(self, future_len=80, action_len=5, agents_len=32, steps=100):
        super().__init__()
        self._agents_len = agents_len
        self._action_len = action_len
        self.noise_level_embedding = nn.Embedding(steps, 256)
        self.decoder = TransformerDecoder(
            future_len, agents_len, self._action_len
        )

    def forward(self, encoder_inputs, noisy_actions, diffusion_step):
        """
        Args:
            noisy_actions: [B, A, T_r, 2], [acc, yaw_rate] Unnormalized actions
            diffusion_step: [B, A]
        Output:
            denoised_states: [B, A, T, 3], [x, y, theta]
        """
        noisy_actions = noisy_actions[:, : self._agents_len]

        if type(diffusion_step) == int:
            diffusion_step = torch.full(
                noisy_actions.shape[:-2],
                diffusion_step,
                dtype=torch.long,
                device=noisy_actions.device,
            )
        else:
            diffusion_step = diffusion_step[:, : self._agents_len]

        current_states = encoder_inputs["agents"][:, : self._agents_len, -1]

        encodings = encoder_inputs["encodings"]
        relations = encoder_inputs["relation_encodings"]

        agents_mask = encoder_inputs["agents_mask"]
        maps_mask = encoder_inputs["maps_mask"]
        traffic_lights_mask = encoder_inputs["traffic_lights_mask"]
        mask = torch.cat([agents_mask, maps_mask, traffic_lights_mask], dim=-1)

        # denoise step
        noise_level = self.noise_level_embedding(diffusion_step)
        noisy_states_local = roll_out(
            current_states,
            noisy_actions,
            action_len=self._action_len,
            global_frame=False,
        )

        denoised_actions_normalized = self.decoder(
            noisy_states_local, noise_level, encodings, relations, mask
        )

        return denoised_actions_normalized

    def reset_agent_length(self, agents_len):
        self._agents_len = agents_len
        self.decoder.reset_agent_length(agents_len)


class AgentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.GRU(8, 256, 2, batch_first=True)
        self.type_embed = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, history, type):
        traj, _ = self.motion(history)
        output = traj[:, -1]  # current frame
        type_embed = self.type_embed(type)
        output = output + type_embed

        return output


class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.traffic_light_embed = nn.Embedding(8, 256)
        self.type_embed = nn.Embedding(21, 256, padding_idx=0)

    def forward(self, inputs):
        # inputs [B, M, W, 5]
        output = self.point(inputs[..., :3])
        output = torch.max(output, dim=-2).values  # max pooling on W

        traffic_light_type = inputs[:, :, 0, 3].long().clamp(0, 7)
        traffic_light_embed = self.traffic_light_embed(traffic_light_type)
        polyline_type = inputs[:, :, 0, 4].long().clamp(0, 20)
        type_embed = self.type_embed(polyline_type)
        output = output + traffic_light_embed + type_embed

        return output


class TrafficLightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_embed = nn.Embedding(8, 256)

    def forward(self, inputs):
        # inputs [B, TL, 3]
        traffic_light_type = inputs[:, :, 2].long().clamp(0, 7)
        type_embed = self.type_embed(traffic_light_type)
        output = type_embed

        return output


class QCMHA(nn.Module):
    """
    Quadratic Complexity Multi-Head Attention module.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.in_proj.bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, query, rel_pos, attn_mask=None):
        """
        Forward pass of the QCMHA module.

        Args:
            query (torch.Tensor): The input query tensor of shape [batch_size, query_length, embed_dim].
            rel_pos (torch.Tensor): The relative position tensor of shape [batch_size, query_length, key_length, embed_dim].
            attn_mask (torch.Tensor, optional): The attention mask tensor of shape [batch_size, query_length, key_length].

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, query_length, embed_dim].
        """
        query = self.in_proj(query)
        b, t, d = query.shape
        query = query.reshape(b, t, self.num_heads, self.head_dim * 3)

        res = torch.split(query, self.head_dim, dim=-1)
        q, k, v = res

        rel_pos_q = rel_pos_v = rel_pos

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        dot_score = torch.matmul(q, k)

        if rel_pos is not None:
            rel_pos_q = rel_pos_q.reshape(
                b, t, t, self.num_heads, self.head_dim
            )
            rel_pos_q = rel_pos_q.permute(0, 3, 1, 4, 2)  # [b, h, q, d, k]
            # [b, h, q, 1, d] * [b, h, q, d, k] -> [b, h, q, 1, k]
            dot_score_rel = torch.matmul(q.unsqueeze(-2), rel_pos_q).squeeze(
                -2
            )
            dot_score += dot_score_rel

        dot_score = dot_score / np.sqrt(self.head_dim)

        if attn_mask is not None:
            dot_score = dot_score - attn_mask.float() * 1e9

        dot_score = F.softmax(dot_score, dim=-1)
        dot_score = self.dropout(dot_score)

        value = torch.matmul(dot_score, v)

        if rel_pos is not None:
            rel_pos_v = rel_pos_v.reshape(
                b, t, t, self.num_heads, self.head_dim
            )
            rel_pos_v = rel_pos_v.permute(0, 3, 1, 2, 4)  # [b, h, q, k, d]
            # [b, h, q, 1, k] * [b, h, q, k, d] -> [b, h, q, d]
            value_rel = torch.matmul(
                dot_score.unsqueeze(-2), rel_pos_v
            ).squeeze(-2)
            value += value_rel

        value = value.permute(0, 2, 1, 3)  # [b, t, h, d//h]
        value = value.reshape(b, t, self.embed_dim)
        value = self.out_proj(value)

        return value


class SelfTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.qc_attention = QCMHA(dim, heads, dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs, relations, mask=None):
        attention_output = self.qc_attention(inputs, relations, mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_freq_bands=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = (
            nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        )

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )

        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, continuous_inputs):
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat(
            [x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1
        )
        x = torch.stack(
            [self.mlps[i](x[:, :, :, i]) for i in range(self.input_dim)]
        ).sum(dim=0)

        return self.to_out(x)


class TransformerEncoder(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.layers = nn.ModuleList([SelfTransformer() for _ in range(layers)])

    def forward(
        self,
        encoded_relations,
        encoded_trajs,
        encoded_polylines,
        encoded_traffic_lights,
        trajs_mask,
        polylines_mask,
        traffic_lights_mask,
    ):
        # relations: [B, N+M+TL, N+M+TL, 256]
        # encoded_trajs: [B, N, 256]
        # encoded_polylines: [B, M, 256]
        # encoded_traffic_lights: [B, TL, 256]

        encodings = torch.cat(
            [encoded_trajs, encoded_polylines, encoded_traffic_lights], dim=1
        )
        encodings_mask = torch.cat(
            [trajs_mask, polylines_mask, traffic_lights_mask], dim=-1
        )
        attention_mask = encodings_mask.unsqueeze(-1).repeat(
            1, 1, encodings_mask.shape[1]
        )
        attention_mask = attention_mask.unsqueeze(1)

        for layer in self.layers:
            encodings = layer(encodings, encoded_relations, attention_mask)

        return encodings


class CrossTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout, batch_first=True
        )
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key, relations, attn_mask=None, key_mask=None):
        # add relations to key and value
        key = key + relations
        value = key

        if key_mask is not None:
            attention_output, _ = self.cross_attention(
                query, key, value, key_padding_mask=key_mask
            )
        elif attn_mask is not None:
            attention_output, _ = self.cross_attention(
                query, key, value, attn_mask=attn_mask
            )
        else:
            attention_output, _ = self.cross_attention(query, key, value)

        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, future_len, agents_len, action_len):
        super().__init__()
        self._future_len = future_len
        self._action_len = action_len
        self._agents_len = agents_len
        self._seq_len = future_len // action_len

        self.time_embedding = nn.Embedding(self._seq_len, 256)
        self.attention_layers = nn.ModuleList(
            [CrossTransformer() for _ in range(4)]
        )
        self.encoder = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.1), nn.Linear(128, 2)
        )

        self.register_buffer("casual_mask", self.generate_casual_mask())
        self.register_buffer("time", torch.arange(self._seq_len).unsqueeze(0))

    def generate_casual_mask(self):
        # Initialize a zero mask
        mask = torch.zeros(
            self._agents_len, self._seq_len, self._agents_len * self._seq_len
        )

        # An agent can attend to all of its own actions
        for i in range(self._agents_len):
            mask[i, :, i * self._seq_len : (i + 1) * self._seq_len] = 1.0

        # An agent can attend to other agents from all previous timesteps but not future timesteps
        for i in range(self._agents_len):
            for j in range(self._agents_len):
                if i != j:
                    for t in range(self._seq_len):
                        mask[
                            i, t, j * self._seq_len : j * self._seq_len + t + 1
                        ] = 1.0

        # Convert to boolean mask
        mask = mask.bool().logical_not()

        return mask

    def forward(
        self, noisy_trajectories, noise_level, encodings, relations, mask
    ):
        """
        noisy_trajectories: [B, Na, T_f, 5]
        """
        # get query
        noisy_trajectories = torch.reshape(
            noisy_trajectories,
            (-1, self._agents_len, self._seq_len, self._action_len, 5),
        )
        future_states = self.encoder(noisy_trajectories)
        future_states = future_states.max(dim=3).values  # [B, Na, T, 256]
        time_embedding = self.time_embedding(self.time)  # [1, T, 256]
        query = future_states + time_embedding[:, None]  # [B, Na, T, 256]
        query = query + noise_level[:, :, None, :]

        # decode denoised actions
        query_content_list = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[0](
                query[:, i],
                query.reshape(-1, self._agents_len * self._seq_len, 256),
                relations[:, i, : self._agents_len].repeat_interleave(
                    self._seq_len, dim=1
                ),
                attn_mask=self.casual_mask[i],
            )  # [B, T, 256]
            query_content = self.attention_layers[1](
                query_content, encodings, relations[:, i], key_mask=mask
            )  # [B, T, 256]
            query_content_list.append(query_content)

        query_content_stack = torch.stack(
            query_content_list, dim=1
        )  # [B, Na, T, 256]
        query_content_stack = query_content_stack + query

        query_content_list = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[2](
                query_content_stack[:, i],
                query_content_stack.reshape(
                    -1, self._agents_len * self._seq_len, 256
                ),
                relations[:, i, : self._agents_len].repeat_interleave(
                    self._seq_len, dim=1
                ),
                attn_mask=self.casual_mask[i],
            )  # [B, T, 256]
            query_content = self.attention_layers[3](
                query_content, encodings, relations[:, i], key_mask=mask
            )  # [B, T, 256]
            query_content_list.append(query_content)

        query_content_stack = torch.stack(
            query_content_list, dim=1
        )  # [B, Na, T, 256]
        actions = self.decoder(query_content_stack)

        return actions

    def reset_agent_length(self, agents_len):
        self._agents_len = agents_len
        new_mask = self.generate_casual_mask().type_as(self.casual_mask)
        self.casual_mask = new_mask
