import pyrallis
from typing import Callable
from pygpudrive.env.config import EnvConfig
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from utils.process import generate_valid_files_json
from algorithms.sb3.ppo.ippo import IPPO
from algorithms.sb3.callbacks import MultiAgentCallback
from algorithms.sb3.wandb_wrapper import WandbLogger, NoWandbLogger, PolicyCheckpointer, NoPolicyCheckpointer
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

    logger = WandbLogger(exp_config, env_config) if exp_config.use_wandb else NoWandbLogger(exp_config, env_config)
    checkpointer = PolicyCheckpointer(logger, exp_config) if exp_config.save_policy else NoPolicyCheckpointer(logger, exp_config)
     
    custom_callback = MultiAgentCallback(
        config=exp_config,
        wandb_logger=logger,
        policy_checkpointer=checkpointer
    )

    # INITIALIZE IPPO
    model = IPPO(
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
        device=exp_config.device,
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
