from madrona_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from madrona_learn.rnn import LSTM, FastLSTM

import math
import torch

def setup_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    partner_obs_tensor = sim.partner_observations_tensor().to_torch()
    room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()

    N, A = self_obs_tensor.shape[0:2]
    batch_size = N * A

    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)

    print(id_tensor.shape)
    print(lidar_tensor.shape)
    
    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
        room_ent_obs_tensor.view(batch_size, *room_ent_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    return obs_tensors, num_obs_features

def process_obs(self_obs, partner_obs, room_ent_obs, lidar, ids):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isnan(partner_obs).any())
    assert(not torch.isnan(room_ent_obs).any())
    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(self_obs).any())
    assert(not torch.isinf(partner_obs).any())
    assert(not torch.isinf(room_ent_obs).any())
    assert(not torch.isinf(lidar).any())

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        partner_obs.view(partner_obs.shape[0], -1),
        room_ent_obs.view(room_ent_obs.shape[0], -1),
        lidar,
        ids,
    ], dim=1)

def make_policy(num_obs_features, num_channels, separate_value):
    encoder = RecurrentBackboneEncoder(
        net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 2,
        ),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    #encoder = BackboneEncoder(
    #    net = MLP(
    #        input_dim = num_obs_features,
    #        num_channels = num_channels,
    #        num_layers = 3,
    #    ),
    #)

    if separate_value:
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = encoder,
            critic_encoder = BackboneEncoder(
                net = MLP(
                    input_dim = num_obs_features,
                    num_channels = num_channels,
                    num_layers = 3,
                ),
            )
        )
    else:
        backbone = BackboneShared(
            process_obs = process_obs_cb,
            encoder = encoder,
        )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [4, 8, 5, 2],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
    )
