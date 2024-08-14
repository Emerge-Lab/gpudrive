from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def sample_discrete_actions(logits: torch.Tensor, action=None):
    categorical_dist = torch.distributions.Categorical(logits=logits)

    if(action is None):
        action = categorical_dist.sample()
    else:
        batch = logits.shape[0]
        action = action.view(batch, -1)
    
    logprob = categorical_dist.log_prob(action)
    entropy = categorical_dist.entropy()

    return action, logprob, entropy
    

def sample_continuous_actions(params: List[torch.Tensor], action=None):
    # Unpack the mean and std deviation from the input list
    mean, log_std = params[0].float(), params[1].float()
    
    # Ensure the tensors are on the same device
    mean = mean
    log_std = log_std

    log_std = torch.clamp(log_std, -100, 2)

    # Convert log_std to std deviation
    std = torch.exp(log_std)
    
    # Create a diagonal covariance matrix from std deviation
    covariance_matrix = torch.diag_embed(std ** 2)
    epsilon = 1e-6
    covariance_matrix += epsilon * torch.eye(covariance_matrix.size(0), device=covariance_matrix.device)
    # Define the Multivariate Gaussian distribution with the given mean and covariance matrix
    normal_dist = MultivariateNormal(mean, covariance_matrix)
    
    if action is None:
        # Sample actions from the distribution using the reparameterization trick
        action = normal_dist.rsample()  # This allows backpropagation through the sampling process
    else:
        batch_size = mean.shape[0]
        # If an action is given, ensure it's used directly (useful for evaluating specific actions' log probs)
        action = action.view(batch_size, -1).to(mean.device)

    # Apply Tanh activation for smooth clamping and then rescale to desired range
    # action[:, 0] = torch.tanh(action[:, 0]) * 4  # Rescale to [-4, 4]
    # action[:, 1] = torch.tanh(action[:, 1]) * 0.6  # Rescale to [-0.6, 0.6]
    assert(mean.shape == action.shape)
    
    # Calculate the log probabilities of the actions
    logprob = normal_dist.log_prob(action)  # No need to sum since MultivariateNormal handles multidimensional cases
    entropy = normal_dist.entropy()
    
    return action, logprob, entropy


class LinearMLP(nn.Module):
    def __init__(self, env, hidden_size=32, output_size=32, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer

        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()

        self.action_space_type = env.unwrapped.action_space_type
        self.num_features = env.unwrapped.num_obs_features
        self.device = env.device

        self.actor = self.build_network(env, hidden_size, output_size, is_actor=True)
        # self.actor = nn.Sequential(
        #     nn.LayerNorm(self.num_features),
        #     layer_init(nn.Linear(np.array(self.num_features).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, env.discrete_action_space.n), std=0.01),
        #     # nn.Softmax(dim=-1)
        # )
        self.critic = self.build_network(env, hidden_size, 1, is_actor=False)
        # self.critic = nn.Sequential(
        #     nn.LayerNorm(self.num_features),
        #     layer_init(nn.Linear(self.num_features, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )

        if self.action_space_type == "continuous":
            self.mean = layer_init(nn.Linear(output_size, env.single_action_space.shape[-1]), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[-1]))

                
    def build_network(self, env, hidden_size, output_size, is_actor):
        layers = [
            nn.LayerNorm(self.num_features),
            layer_init(nn.Linear(self.num_features, hidden_size)),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            layer_init(nn.Linear(hidden_size, output_size)),
            nn.Tanh()
        ]
        if is_actor:
            if self.action_space_type == "discrete":
                layers.append(layer_init(nn.Linear(output_size, env.discrete_action_space.n), std=0.01))
                # layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)
    
    def get_actor_logits(self, x) -> Union[torch.Tensor, List[torch.Tensor]]:
        action = self.actor(x)
        if self.action_space_type == "discrete":
            return action
        else:
            mean = self.mean(action)
            mean[:,0] = torch.tanh(mean[:,0]) * 4
            mean[:,1] = torch.tanh(mean[:,1]) * 0.6
            return [mean, self.log_std]

    def get_value(self, x, state=None):
        value = self.critic(x)
        return value

    def get_action_and_value(self, x, action=None):
        x = x.to(self.device)
        logits = self.get_actor_logits(x)
        if self.action_space_type == "discrete":
            action, logprob, entropy = sample_discrete_actions(logits, action) # type: ignore
        else:
            action, logprob, entropy = sample_continuous_actions(logits, action) # type: ignore
        value = self.critic(x)
        return action, logprob, entropy, value

    def forward(self, env_outputs, action=None):
        '''Forward pass for PufferLib compatibility'''
        
        return self.get_action_and_value(env_outputs, action)
    

class MultiHeadLinear(nn.Module):
    def __init__(self, env, input_size = 64, hidden_size=32, output_size=32, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer

        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()

        self.action_space_type = env.unwrapped.action_space_type
        self.num_features = env.unwrapped.num_obs_features
        self.self_obs = env.unwrapped.self_obs_shape
        self.partner_obs = env.unwrapped.partner_obs_shape
        self.map_obs = env.unwrapped.map_obs_shape
        self.device = env.device

        self.self_obs_embed = nn.Sequential(
            nn.LayerNorm(self.self_obs),
            layer_init(nn.Linear(self.self_obs, input_size)),
            nn.Tanh()
        )

        self.partner_obs_embed = nn.Sequential(
            nn.LayerNorm(self.partner_obs),
            layer_init(nn.Linear(self.partner_obs, input_size)),
            nn.Tanh()
        )

        self.map_obs_embed = nn.Sequential(
            nn.LayerNorm(self.map_obs),
            layer_init(nn.Linear(self.map_obs, input_size)),
            nn.Tanh()
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(input_size*3),
            layer_init(nn.Linear(input_size*3, hidden_size)),
            nn.Tanh()
        )
        
        self.critic = self.build_network(env, hidden_size, 1, is_actor=False)

        if self.action_space_type == "discrete":
            self.actor = self.build_network(env, hidden_size, hidden_size, is_actor=True)

        if self.action_space_type == "continuous":
            self.mean = layer_init(nn.Linear(output_size, env.single_action_space.shape[-1]), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[-1]))

                
    def build_network(self, env, hidden_size, output_size, is_actor):
        layers = [
            layer_init(nn.Linear(hidden_size, output_size, bias=True)),
            nn.Tanh()
        ]
        if is_actor:
            if self.action_space_type == "discrete":
                layers.append(layer_init(nn.Linear(output_size, env.discrete_action_space.n), std=0.01))
                # layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)
    
    def get_actor_logits(self, x) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.action_space_type == "discrete":
            return self.actor(x)
        else:
            mean = self.mean(x)
            return [mean, self.log_std]

    def get_value(self, x, state=None):
        value = self.critic(x)
        return value

    def get_action_and_value(self, x, action=None):
        x = x.to(self.device)
        logits = self.get_actor_logits(x)
        if self.action_space_type == "discrete":
            action, logprob, entropy = sample_discrete_actions(logits, action) # type: ignore
        else:
            action, logprob, entropy = sample_continuous_actions(logits, action) # type: ignore
        value = self.critic(x)
        return action, logprob, entropy, value

    def forward(self, env_outputs, action=None):
        '''Forward pass for PufferLib compatibility'''
        obs_embed = self.self_obs_embed(env_outputs[:,:self.self_obs])
        partner_embed = self.partner_obs_embed(env_outputs[:,self.self_obs:self.partner_obs+self.self_obs])
        map_embed = self.map_obs_embed(env_outputs[:,self.self_obs+self.partner_obs:])
        proj = self.proj(torch.cat([obs_embed, partner_embed, map_embed], dim=-1))
        return self.get_action_and_value(proj, action)