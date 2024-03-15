import torch
import wandb

import sys
#TODO: Fix
sys.path.append('/home/emerge/gpudrive')

# Import env wrapper that makes the GPU gym env compatible with stable-baselines3
from env.sb3_wrapper import SB3MultiAgentEnv

from callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.mappo import MAPPO

if __name__ == "__main__":
    
    # Make SB3-compatible environment
    env = SB3MultiAgentEnv(
        num_worlds=1, 
        max_cont_agents=1, 
        data_dir='waymo_data', 
        device='cuda', 
        auto_reset=True
    )
    
    # Initialize wandb    
    wandb.login()
    run = wandb.init(
        project="gpu_drive",
        group="single_agent",
        sync_tensorboard=True,
    )
    run_id = run.id
    
    # Initialize custom callback
    custom_callback = MultiAgentCallback(
        wandb_run=run if run_id is not None else None,
    )
    
    model = MAPPO(      
        policy="MlpPolicy", # Policy type
        n_steps=2048, # Number of steps per rollout
        batch_size=256, # Minibatch size
        env=env, # Our wrapped environment
        seed=42, # Always seed for reproducibility
        verbose=1,
        tensorboard_log=f"runs/{run_id}" if run_id is not None else None, # Sync with wandb
    )
    
    # Learn
    model.learn(
        total_timesteps=2_000_000,
        callback=custom_callback,
    )