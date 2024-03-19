import pandas as pd
import logging

from time import perf_counter
# Import env wrapper that makes the GPU gym env compatible with stable-baselines3
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from algorithms.ppo.sb3.callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.mappo import MAPPO


if __name__ == "__main__":
    
    # Settings
    TOTAL_STEPS = 10_000
    MAX_CONT_AGENTS = 1
    NUM_BENCH_WORLDS = [1, 10, 100] 
    DATA_DIR = ['waymo_data', 'data_10', 'data_100']    
    
    # Storage 
    df = pd.DataFrame(columns=['num_worlds', 'num_frames', 'training time (s)', 'fps'])
    idx = 0
    
    for data_dir, num_worlds in zip(DATA_DIR, NUM_BENCH_WORLDS):
        
        logging.info(f"Running PPO benchmark with {num_worlds} worlds.")
        
        ppo_start = perf_counter()
        
        # Make SB3-compatible environment
        env = SB3MultiAgentEnv(
            num_worlds=num_worlds, 
            max_cont_agents=MAX_CONT_AGENTS, 
            data_dir=data_dir,
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
        model.learn(total_timesteps=TOTAL_STEPS)
        
        ppo_end = perf_counter()
        
        # Add results to dataframe
        res = {
            'num_worlds': num_worlds,
            'num_frames': model.num_timesteps,
            'time': ppo_end - ppo_start,
            'fps': model.num_timesteps / (ppo_end - ppo_start),
            'num_cont_agents': MAX_CONT_AGENTS,
        }
        
        df.loc[idx] = res
        
        idx += 1
        # Checkpointing
        df.to_csv('ppo_sim_benchmark.csv')
        
        
        env.close()