"""Obtain a policy using behavioral cloning."""

# Torch
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

import wandb

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from algorithms.il.data_generation import generate_state_action_pairs
from baselines.il.config import BehavCloningConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":

    # Configurations
    env_config = EnvConfig(use_bicycle_model=True)
    render_config = RenderConfig()
    bc_config = BehavCloningConfig()

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_dir=bc_config.data_dir,
        render_config=render_config,
        num_worlds=bc_config.num_worlds,
        max_cont_agents=bc_config.max_cont_agents,
        device=bc_config.device,  # Use DEVICE here for consistency
    )

    # Generate expert actions and observations
    expert_obs, expert_actions = generate_state_action_pairs(
        env=env,
        device=bc_config.device,
        discretize_actions=bc_config.discretize_actions,
        use_action_indices=bc_config.use_action_indices,
        make_video=bc_config.make_sanity_check_video,
    )

    run = wandb.init(
        mode=bc_config.wandb_mode,
        project=bc_config.wandb_project,
    )

    class ExpertDataset(torch.utils.data.Dataset):
        def __init__(self, obs, actions):
            self.obs = obs
            self.actions = actions

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, idx):
            return self.obs[idx], self.actions[idx]

    # Make dataloader
    expert_dataset = ExpertDataset(expert_obs, expert_actions)
    expert_data_loader = DataLoader(
        expert_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )

    # Define network
    class FeedForward(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(FeedForward, self).__init__()
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

    # Build model
    bc_policy = FeedForward(
        input_size=env.observation_space.shape[0],
        hidden_size=bc_config.hidden_size,
        output_size=env.action_space.n,
    ).to(bc_config.device)

    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=bc_config.lr)

    global_step = 0
    for epoch in range(bc_config.epochs):
        for i, (obs, expert_action) in enumerate(expert_data_loader):

            obs, expert_action = obs.to(bc_config.device), expert_action.to(
                bc_config.device
            )

            # Forward pass
            log_prob = bc_policy._log_prob(obs, expert_action)
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
            
    
    # Save policy
    if bc_config.save_model:
        torch.save(bc_policy, f"{bc_config.model_path}/bc_policy.pt")