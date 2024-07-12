"""Obtain a policy using behavioral cloning."""

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

import wandb

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from data_generation import generate_state_action_pairs


BATCH_SIZE = 512
EPOCHS = 800
MINIBATCHES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # Settings
    env_config = EnvConfig(sample_method="pad_n", use_bicycle_model=True)
    render_config = RenderConfig()

    # Generate expert actions and observations
    expert_obs, expert_actions = generate_state_action_pairs(
        env_config=env_config,
        render_config=render_config,
        max_num_objects=128,
        num_worlds=10,
        data_dir="example_data",
        device="cuda",
        discretize_actions=False,  # Discretize the expert actions
        make_video=True,  # Record the trajectories as sanity check
        save_path="output_video.mp4",
    )

    # Prepare dataset and data loader using the torch dataset and data loader
    class ExpertDataset(torch.utils.data.Dataset):
        def __init__(self, obs, actions):
            self.obs = obs
            self.actions = actions

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, idx):
            return self.obs[idx], self.actions[idx]

    expert_dataset = ExpertDataset(expert_obs, expert_actions)
    expert_data_loader = DataLoader(
        expert_dataset, batch_size=32, shuffle=True
    )

    # Define network
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Net, self).__init__()
            self.nn = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size)])

        def dist(self, obs):
            """Generate action distribution."""
            x_out = self.nn(obs.float())
            return [Categorical(logits=head(x_out)) for head in self.heads]

        def forward(self, obs, deterministic=False):
            """Generate an output from tensor input."""
            action_dist = self.dist(obs)

            if deterministic:
                actions_idx = action_dist[0].logits.argmax(axis=-1)
            else:
                actions_idx = action_dist.sample()
            return actions_idx

        def _log_prob(self, obs, expert_actions):
            pred_action_dist = self.dist(obs)
            log_prob = pred_action_dist[0].log_prob(expert_actions).mean()
            return log_prob

    # == TODO ==
    # Build model
    bc_policy = Net(
        input_size=env.observation_space.shape[0],
        hidden_size=800,
        output_size=waymo_iterator.action_space.n,
    ).to(DEVICE)

    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=1e-4)

    global_step = 0
    for epoch in range(EPOCHS):
        for i in range(MINIBATCHES):

            # Get batch of obs-act pairs
            obs, expert_action = next(expert_data_loader)
            obs, expert_action = obs.to(DEVICE), expert_action.to(DEVICE)

            # Forward pass
            log_prob = bc_policy._log_prob(obs, expert_action.float())
            loss = -log_prob

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_action = bc_policy(obs, deterministic=True)
                accuracy = (
                    expert_action == pred_action
                ).sum() / expert_action.shape[0]

            wandb.log(
                {
                    "global_step": global_step,
                    "loss": loss.item(),
                    "acc": accuracy,
                }
            )

            global_step += 1
