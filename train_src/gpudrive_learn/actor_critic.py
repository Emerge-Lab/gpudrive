import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .action import DiscreteActionDistributions
from .profile import profile

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def _flatten_obs_sequence(self, obs):
        return [o.view(-1, *o.shape[2:]) for o in obs]

    def forward(self, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        raise NotImplementedError


class DiscreteActor(nn.Module):
    def __init__(self, actions_num_buckets, impl):
        super().__init__()

        self.actions_num_buckets = actions_num_buckets
        self.impl = impl

    def forward(self, features_in):
        logits = self.impl(features_in)

        return DiscreteActionDistributions(
                self.actions_num_buckets, logits=logits)


class Critic(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl 

    def forward(self, features_in):
        return self.impl(features_in)


class RecurrentStateConfig:
    def __init__(self, shapes):
        self.shapes = shapes
    

class ActorCritic(nn.Module):
    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.recurrent_cfg = backbone.recurrent_cfg
        self.actor = actor
        self.critic = critic

    # Direct call intended for debugging only, should use below
    # specialized functions
    def forward(self, rnn_states, *obs):
        actor_features, critic_features, new_rnn_states = self.backbone(
            rnn_states, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        return action_dists, values, new_rnn_states

    def fwd_actor(self, actions_out, rnn_states_out, rnn_states_in, *obs_in):
        actor_features = self.backbone.fwd_actor_only(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        action_dists.best(out=actions_out)

    def fwd_critic(self, values_out, rnn_states_out, rnn_states_in, *obs_in):
        features = self.backbone.fwd_critic_only(
            rnn_states_out, rnn_states_in, *obs_in)
        values_out[...] = self.critic(features)

    def fwd_rollout(self, actions_out, log_probs_out, values_out,
                      rnn_states_out, rnn_states_in, *obs_in):
        actor_features, critic_features = self.backbone.fwd_rollout(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        action_dists.sample(actions_out, log_probs_out)
        values_out[...] = values

    def fwd_update(self, rnn_states, sequence_breaks, rollout_actions, *obs):
        actor_features, critic_features = self.backbone.fwd_sequence(
            rnn_states, sequence_breaks, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(
            T * N, *rollout_actions.shape[2:])

        log_probs, entropies = action_dists.action_stats(flattened_actions)

        log_probs = log_probs.view(T, N, *log_probs.shape[1:])
        entropies = entropies.view(T, N, *entropies.shape[1:])
        values = values.view(T, N, *values.shape[1:])

        return log_probs, entropies, values

class BackboneEncoder(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.rnn_state_shape = None

    def forward(self, rnn_states, *inputs):
        features = self.net(*inputs)
        return features, None

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        return self.net(*inputs)

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states,
                     sequence_breaks, *flattened_inputs):
        return self.net(*flattened_inputs)

class RecurrentBackboneEncoder(nn.Module):
    def __init__(self, net, rnn):
        super().__init__()
        self.net = net
        self.rnn = rnn
        self.rnn_state_shape = rnn.hidden_shape

    def forward(self, rnn_states_in, *inputs):
        features = self.net(*inputs)

        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        return rnn_out, new_rnn_states

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        features = self.net(*inputs)
        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        # FIXME: proper inplace
        if rnn_states_out != None:
            rnn_states_out[...] = rnn_states_in

        return rnn_out

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states, sequence_breaks,
                     *flattened_inputs):
        features = self.net(*flattened_inputs)
        features_seq = features.view(
            *sequence_breaks.shape[0:2], *features.shape[1:])

        with profile('rnn.fwd_sequence'):
            rnn_out_seq = self.rnn.fwd_sequence(
                features_seq, rnn_start_states, sequence_breaks)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])
        return rnn_out_flattened

class BackboneShared(Backbone):
    def __init__(self, process_obs, encoder):
        super().__init__()
        self.process_obs = process_obs
        self.encoder = encoder 

        if encoder.rnn_state_shape:
            self.recurrent_cfg = RecurrentStateConfig([encoder.rnn_state_shape])
            self.extract_rnn_state = lambda x: x[0] if x != None else None
            self.package_rnn_state = lambda x: (x,)
        else:
            self.recurrent_cfg = RecurrentStateConfig([])
            self.extract_rnn_state = lambda x: None
            self.package_rnn_state = lambda x: ()

    def forward(self, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        features, new_rnn_states = self.encoder(
            self.extract_rnn_state(rnn_states_in), processed_obs)
        return features, features, self.package_rnn_state(new_rnn_states)

    def _rollout_common(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder.fwd_inplace(
            self.extract_rnn_state(rnn_states_out),
            self.extract_rnn_state(rnn_states_in),
            processed_obs,
        )

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        features = self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

        return features, features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        features = self.encoder.fwd_sequence(
            self.extract_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return features, features


class BackboneSeparate(Backbone):
    def __init__(self, process_obs, actor_encoder, critic_encoder):
        super().__init__()
        self.process_obs = process_obs
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder

        rnn_state_shapes = []

        if actor_encoder.rnn_state_shape == None:
            self.extract_actor_rnn_state = lambda rnn_states: None
        else:
            actor_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(actor_encoder.rnn_state_shape)
            self.extract_actor_rnn_state = \
                lambda rnn_states: rnn_states[actor_rnn_idx]

        if critic_encoder.rnn_state_shape == None:
            self.extract_critic_rnn_state = lambda rnn_states: None
        else:
            critic_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(critic_encoder.rnn_state_shape)
            self.extract_critic_rnn_state = \
                lambda rnn_states: rnn_states[critic_rnn_idx]

        if (actor_encoder.rnn_state_shape and
                critic_encoder.rnn_state_shape):
            self.package_rnn_states = lambda a, c: (a, c)
        elif actor_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (a,)
        elif critic_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (c,)
        else:
            self.package_rnn_states = lambda a, c: ()

        self.recurrent_cfg = RecurrentStateConfig(rnn_state_shapes)

    def forward(self, rnn_states, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features, new_actor_rnn_states = self.actor_encoder(
            self.extract_actor_rnn_state(rnn_states),
            processed_obs)
        critic_features, new_critic_rnn_states = self.critic_encoder(
            self.extract_critic_rnn_state(rnn_states),
            processed_obs)

        return actor_features, critic_features, self.package_rnn_states(
            new_actor_rnn_states, new_critic_rnn_states)

    def _rollout_common(self, rnn_states_out, rnn_states_in,
                        *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder.fwd_inplace(
            rnn_states_out, rnn_states_in, processed_obs)

    def fwd_actor_only(self, rnn_states_out, rnn_states_in,
                       *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.actor_encoder.fwd_inplace(
            self.extract_actor_rnn_state(rnn_states_out) if rnn_states_out else None,
            self.extract_actor_rnn_state(rnn_states_in),
            processed_obs)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in,
                        *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.critic_encoder.fwd_inplace(
            self.extract_critic_rnn_state(rnn_states_out) if rnn_states_out else None,
            self.extract_critic_rnn_state(rnn_states_in),
            processed_obs)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features = self.actor_encoder.fwd_inplace(
            self.extract_actor_rnn_state(rnn_states_out),
            self.extract_actor_rnn_state(rnn_states_in),
            processed_obs)

        critic_features = self.critic_encoder.fwd_inplace(
            self.extract_critic_rnn_state(rnn_states_out),
            self.extract_critic_rnn_state(rnn_states_in),
            processed_obs)

        return actor_features, critic_features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        actor_features = self.actor_encoder.fwd_sequence(
            self.extract_actor_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        critic_features = self.critic_encoder.fwd_sequence(
            self.extract_critic_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return actor_features, critic_features
