from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


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
    mean = mean.to(mean.device)
    log_std = log_std.to(mean.device)
    
    # Convert log_std to std deviation
    std = torch.exp(log_std)
    
    # Create a diagonal covariance matrix from std deviation
    covariance_matrix = torch.diag_embed(std ** 2)
    
    # Define the Multivariate Gaussian distribution with the given mean and covariance matrix
    normal_dist = MultivariateNormal(mean, covariance_matrix)
    
    if action is None:
        # Sample actions from the distribution using the reparameterization trick
        action = normal_dist.rsample()  # This allows backpropagation through the sampling process
    else:
        batch_size = mean.shape[0]
        # If an action is given, ensure it's used directly (useful for evaluating specific actions' log probs)
        action = action.view(batch_size, -1).to(mean.device)

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

        self.action_space_type = env.action_space_type
        self.num_features = env.num_obs_features
        self.device = env.device

        self.actor = self.build_network(env, hidden_size, output_size, is_actor=True).to(self.device)
        self.critic = self.build_network(env, hidden_size, 1, is_actor=False).to(self.device)

        if self.action_space_type == "continuous":
            self.mean = nn.Linear(output_size, env.single_action_space.shape[-1]).to(self.device)
            self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[-1])).to(self.device)

    def build_network(self, env, hidden_size, output_size, is_actor):
        layers = [
            nn.Linear(self.num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        ]
        if is_actor:
            if self.action_space_type == "discrete":
                layers.append(nn.Linear(output_size, env.action_space.shape[-1]))
                layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)
    
    def get_actor_logits(self, x) -> Union[torch.Tensor, List[torch.Tensor]]:
        action = self.actor(x)
        if self.action_space_type == "discrete":
            return action
        else:
            mean = self.mean(action)
            return [mean, self.log_std]

    def get_value(self, x, state=None):
        value = self.critic(x)
        return value

    def get_action_and_value(self, x, action=None):
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