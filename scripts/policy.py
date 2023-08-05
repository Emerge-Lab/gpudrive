from madrona_escape_room_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_escape_room_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from madrona_escape_room_learn.rnn import LSTM

import math
import torch

def setup_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    partner_obs_tensor = sim.partner_observations_tensor().to_torch()
    room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
    door_obs_tensor = sim.door_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

    N, A = self_obs_tensor.shape[0:2]
    batch_size = N * A

    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)

    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
        room_ent_obs_tensor.view(batch_size, *room_ent_obs_tensor.shape[2:]),
        door_obs_tensor.view(batch_size, *door_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    return obs_tensors, num_obs_features

def process_obs(self_obs, partner_obs, room_ent_obs,
                door_obs, lidar, steps_remaining, ids):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(partner_obs).any())
    assert(not torch.isinf(partner_obs).any())

    assert(not torch.isnan(room_ent_obs).any())
    assert(not torch.isinf(room_ent_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        partner_obs.view(partner_obs.shape[0], -1),
        room_ent_obs.view(room_ent_obs.shape[0], -1),
        door_obs.view(door_obs.shape[0], -1),
        lidar.view(lidar.shape[0], -1),
        steps_remaining.float() / 200,
        ids,
    ], dim=1)

def make_policy(num_obs_features, num_channels, separate_value):
    #encoder = RecurrentBackboneEncoder(
    #    net = MLP(
    #        input_dim = num_obs_features,
    #        num_channels = num_channels,
    #        num_layers = 2,
    #    ),
    #    rnn = LSTM(
    #        in_channels = num_channels,
    #        hidden_channels = num_channels,
    #        num_layers = 1,
    #    ),
    #)

    encoder = BackboneEncoder(
        net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 3,
        ),
    )

    if separate_value:
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = encoder,
            critic_encoder = RecurrentBackboneEncoder(
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
        )
    else:
        backbone = BackboneShared(
            process_obs = process_obs,
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
