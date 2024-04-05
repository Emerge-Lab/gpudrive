import wandb

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig

# Import env wrapper that makes the GPU gym env compatible with stable-baselines3
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from algorithms.ppo.sb3.callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.mappo import MAPPO

if __name__ == "__main__":

    config = EnvConfig()
    
    # Make SB3-compatible environment
    env = SB3MultiAgentEnv(
        config=config,
        num_worlds=1,
        max_cont_agents=1,
        data_dir="waymo_data",
        device="cuda",
        auto_reset=True,
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
        policy="MlpPolicy",  # Policy type
        n_steps=100,  # Number of steps per rollout
        batch_size=100,  # Minibatch size
        env=env,  # Our wrapped environment
        seed=42,  # Always seed for reproducibility
        verbose=1,
        tensorboard_log=f"runs/{run_id}"
        if run_id is not None
        else None,  # Sync with wandb
    )

    # Learn
    model.learn(
        total_timesteps=1000,
        callback=custom_callback,
    )
