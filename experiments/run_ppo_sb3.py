import wandb

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig

# Import env wrapper that makes gym env compatible with stable-baselines3
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from algorithms.ppo.sb3.callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.mappo import MAPPO

if __name__ == "__main__":

    config = EnvConfig()

    # Make SB3-compatible environment
    env = SB3MultiAgentEnv(
        config=config,
        num_worlds=3,
        max_cont_agents=4,
        data_dir="waymo_data",
        device="cuda",
    )

    # Initialize wandb
    wandb.login()
    run = wandb.init(
        project="please_drive",
        group="new_roadlines_test",
        sync_tensorboard=True,
    )
    run_id = run.id

    # Initialize custom callback
    custom_callback = MultiAgentCallback(
        wandb_run=run if run_id is not None else None,
    )

    model = MAPPO(
        policy="MlpPolicy",  # Policy type
        n_steps=2048,  # Number of steps per rollout
        batch_size=256,  # Minibatch size
        env=env,  # Our wrapped environment
        seed=42,  # Always seed for reproducibility
        verbose=0,
        tensorboard_log=f"runs/{run_id}"
        if run_id is not None
        else None,  # Sync with wandb
    )

    # Learn
    model.learn(
        total_timesteps=3_000_000,
        callback=custom_callback,
    )

    run.finish()
