import wandb

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig

# Import env wrapper that makes gym env compatible with stable-baselines3
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from algorithms.ppo.sb3.callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.mappo import MAPPO

from baselines.config import ExperimentConfig

if __name__ == "__main__":

    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        normalize_obs=False,
    )

    exp_config = ExperimentConfig(
        render=True,
    )

    # Make SB3-compatible environment
    env = SB3MultiAgentEnv(
        config=env_config,
        num_worlds=5,
        max_cont_agents=2,
        data_dir="waymo_data",
        device="cuda",
    )

    run = wandb.init(
        project="rl_benchmarking",
        sync_tensorboard=True,
    )
    run_id = run.id

    # Initialize custom callback
    custom_callback = MultiAgentCallback(
        config=exp_config,
        wandb_run=run if run_id is not None else None,
    )

    model = MAPPO(
        policy=exp_config.policy,
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
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
    env.close()
