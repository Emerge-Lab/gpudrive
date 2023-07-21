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
    pos_obs_tensor = sim.position_observation_tensor().to_torch()
    to_others_tensor = sim.to_other_agents_tensor().to_torch()
    to_buttons_tensor = sim.to_buttons_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    
    obs_tensors = [
        pos_obs_tensor,
        to_others_tensor,
        to_buttons_tensor,
        lidar_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    def process_obs(pos_obs, to_others, to_buttons, lidar):
        assert(not torch.isnan(pos_obs).any())
        assert(not torch.isnan(to_others).any())
        assert(not torch.isnan(to_buttons).any())
        assert(not torch.isnan(lidar).any())
        assert(not torch.isinf(pos_obs).any())
        assert(not torch.isinf(to_others).any())
        assert(not torch.isinf(to_buttons).any())
        assert(not torch.isinf(lidar).any())

        return torch.cat([
            pos_obs.view(pos_obs.shape[0], -1),
            to_others.view(to_others.shape[0], -1),
            to_buttons.view(to_buttons.shape[0], -1),
            lidar
        ], dim=1)

    return obs_tensors, process_obs, num_obs_features

def make_policy(process_obs_cb, num_obs_features, num_channels, separate_value):
    move_action_dim = 5
    
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

    if separate_value:
        backbone = BackboneSeparate(
            process_obs = process_obs_cb,
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
            process_obs = process_obs_cb,
            encoder = encoder,
        )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [move_action_dim, move_action_dim, move_action_dim], num_channels),
        critic = LinearLayerCritic(num_channels),
    )
