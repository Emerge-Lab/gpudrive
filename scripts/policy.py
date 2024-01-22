from gpudrive_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from gpudrive_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from gpudrive_learn.rnn import LSTM

import math
import torch

def setup_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    partner_obs_tensor = sim.partner_observations_tensor().to_torch()
    map_obs_tensor = sim.map_observation_tensor().to_torch()
    # lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

    N, A, O = self_obs_tensor.shape[0:3] # N = num worlds, A = num agents, O = num obs features
    batch_size = N * A

    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, A).expand(N, A).reshape(batch_size, 1)

    # Flatten map obs tensor of shape (N, R, 4) to (N, 4 * R)
    map_obs_tensor = map_obs_tensor.view(N, map_obs_tensor.shape[1]*map_obs_tensor.shape[2])
    # map_obs_tensor = map_obs_tensor.view(N, 1, -1).expand(N, A, -1).reshape(batch_size, -1)
    map_obs_tensor = map_obs_tensor.repeat(N*A//N, 1)

    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
        map_obs_tensor.view(batch_size, *map_obs_tensor.shape[1:]),
        # lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    return obs_tensors, num_obs_features

def process_obs(self_obs, partner_obs, map_obs, steps_remaining, ids):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(partner_obs).any())
    assert(not torch.isinf(partner_obs).any())

    # assert(not torch.isnan(lidar).any())
    # assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        partner_obs.view(partner_obs.shape[0], -1),
        map_obs.view(map_obs.shape[0], -1),
        # lidar.view(lidar.shape[0], -1),
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
            [4, 4, 4],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
    )
