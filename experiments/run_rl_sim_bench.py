import numpy as np
import wandb

# Import env wrapper that makes the GPU gym env compatible with stable-baselines3
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from algorithms.ppo.sb3.callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.mappo import MAPPO


if __name__ == "__main__":
    
    MAX_CONT_AGENTS = 1
    NUM_BENCH_WORLDS = [5, 10] #np.arange(1, 101, 10)
    fps_array = []
    
    for i, num_worlds in enumerate(NUM_BENCH_WORLDS):
        
        # Make SB3-compatible environment
        env = SB3MultiAgentEnv(
            num_worlds=2, 
            max_cont_agents=MAX_CONT_AGENTS, 
            data_dir='formatted_json_v2_no_tl_train', 
            device='cuda', 
            auto_reset=True
        )
        
        model = MAPPO(      
            policy="MlpPolicy", # Policy type
            n_steps=2048, # Number of steps per rollout
            batch_size=256, # Minibatch size
            env=env, # Our wrapped environment
            seed=42, # Always seed for reproducibility
            verbose=1,
        )
        
        # Learn
        model.learn(total_timesteps=10_000)