from typing import List, Any
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
import torch
from abc import ABC, abstractmethod

# GPU Drive Gym Environment
from gpudrive_gym_env import Env
import wandb

from wrappers import SB3EnvWrapper
from stable_baselines3 import PPO

if __name__ == "__main__":
    
    # Make and wrap environment
    env = Env(num_worlds=1, max_cont_agents=1, device='cuda')
    env = SB3EnvWrapper(env)
    
    # # Initialize wandb    
    # wandb.login()
    # run = wandb.init(
    #     project="gpu_drive",
    #     group="single_agent",
    #     sync_tensorboard=True,
    # )
    # run_id = run.id
    
    model = PPO(      
        policy="MlpPolicy",  # Policy type
        n_steps=4096, # Number of steps per rollout
        batch_size=256, # Minibatch size
        env=env, # Our wrapped environment
        seed=42, # Always seed for reproducibility
        verbose=0,
        #tensorboard_log=f"runs/{run_id}" if run_id is not None else None, # Sync with wandb
    )