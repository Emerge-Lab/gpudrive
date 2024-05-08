import wandb
import torch
import torch

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig

# Import env wrapper that makes gym env compatible with stable-baselines3
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from algorithms.ppo.sb3.callbacks import MultiAgentCallback

# Import adapted PPO version
from algorithms.ppo.sb3.ippo import IPPO

from baselines.config import ExperimentConfig

torch.cuda.empty_cache()

if __name__ == "__main__":

    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=False,
        sample_method="rand_n",
    )

    exp_config = ExperimentConfig(
        render=True,
    )

    # Make SB3-compatible environment
    env = SB3MultiAgentEnv(
        config=env_config,
        num_worlds=2,
        max_cont_agents=128,
        data_dir="formatted_json_v2_no_tl_train",
        device=exp_config.device,
    )

    run = wandb.init(
        project="rl_bench",
        group="render_test",
        sync_tensorboard=True,
    )
    run_id = run.id

    # Initialize custom callback
    custom_callback = MultiAgentCallback(
        config=exp_config,
        wandb_run=run if run_id is not None else None,
    )

    model = IPPO(
        policy=exp_config.policy,
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
        device=exp_config.device,
        tensorboard_log=f"runs/{run_id}"
        if run_id is not None
        else None,  # Sync with wandb
    )

    # Learn
    model.learn(
        total_timesteps=exp_config.total_timesteps,
        callback=custom_callback,
    )

    run.finish()
    env.close()
