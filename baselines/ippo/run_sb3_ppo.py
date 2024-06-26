import wandb
import pyrallis
from typing import Callable
from pygpudrive.env.config import EnvConfig
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from utils.process import generate_valid_files_json
from algorithms.sb3.ppo.ippo import IPPO
from algorithms.sb3.callbacks import MultiAgentCallback
from baselines.ippo.config import ExperimentConfig


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(exp_config: ExperimentConfig):
    """Run PPO training with stable-baselines3."""

    # CONFIG
    env_config = EnvConfig()

    # MAKE SB3-COMPATIBLE ENVIRONMENT
    env = SB3MultiAgentEnv(
        config=env_config,
        num_worlds=exp_config.num_worlds,
        max_cont_agents=env_config.num_controlled_vehicles,
        data_dir=exp_config.data_dir,
        device=exp_config.device,
    )

    # INIT WANDB
    run_id = None
    if exp_config.use_wandb:
        run = wandb.init(
            project=exp_config.project_name,
            group=exp_config.group_name,
            sync_tensorboard=exp_config.sync_tensorboard,
            tags=exp_config.tags,
            mode=exp_config.wandb_mode,
            config={**exp_config.__dict__, **env_config.__dict__},
        )
        run_id = run.id

    # CALLBACK
    custom_callback = MultiAgentCallback(
        config=exp_config,
        wandb_run=run if run_id is not None else None,
    )

    # INITIALIZE IPPO
    model = IPPO(
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
        device=exp_config.device,
        tensorboard_log=f"runs/{run_id}"
        if run_id is not None
        else None,  # Sync with wandb
        mlp_class=exp_config.mlp_class,
        policy=exp_config.policy,
        gamma=exp_config.gamma,
        gae_lambda=exp_config.gae_lambda,
        vf_coef=exp_config.vf_coef,
        clip_range=exp_config.clip_range,
        learning_rate=linear_schedule(exp_config.lr),
        ent_coef=exp_config.ent_coef,
        n_epochs=exp_config.n_epochs,
        env_config=env_config,
        exp_config=exp_config,
    )

    # LEARN
    model.learn(
        total_timesteps=exp_config.total_timesteps,
        callback=custom_callback,
    )

    run.finish()
    env.close()


if __name__ == "__main__":

    exp_config = pyrallis.parse(config_class=ExperimentConfig)

    if exp_config.generate_valid_json:
        actual_num_files = generate_valid_files_json(
            num_unique_scenes=exp_config.train_on_k_unique_scenes,
            data_dir=exp_config.data_dir,
        )
        
    exp_config.actual_num_files = actual_num_files

    train(exp_config)
