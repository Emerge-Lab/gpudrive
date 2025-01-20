"""
This implementation is adapted from the demo in PufferLib by Joseph Suarez,
which in turn is adapted from Costa Huang's CleanRL PPO + LSTM implementation.
Links
- PufferLib: https://github.com/PufferAI/PufferLib/blob/dev/demo.py
- Cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import os
from typing import Optional
from typing_extensions import Annotated
import yaml
from datetime import datetime
import torch
import numpy as np
import wandb
from box import Box
import time
import random

from integrations.rl.puffer import ppo
from integrations.rl.puffer.puffer_env import env_creator

from networks.late_fusion import LateFusionTransformer
from pygpudrive.env.dataset import SceneDataLoader

import pufferlib
import pufferlib.vector
import pufferlib.frameworks.cleanrl
from rich.console import Console

import typer
from typer import Typer

app = Typer()


def log_normal(mean, scale, clip):
    """Samples normally spaced points on a log 10 scale.
    mean: Your center sample point
    scale: standard deviation in base 10 orders of magnitude
    clip: maximum standard deviations

    Example: mean=0.001, scale=1, clip=2 will produce data from
    0.1 to 0.00001 with most of it between 0.01 and 0.0001
    """
    return 10 ** np.clip(
        np.random.normal(
            np.log10(mean),
            scale,
        ),
        a_min=np.log10(mean) - clip,
        a_max=np.log10(mean) + clip,
    )


def logit_normal(mean, scale, clip):
    """log normal but for logit data like gamma and gae_lambda"""
    return 1 - log_normal(1 - mean, scale, clip)


def uniform_pow2(min, max):
    """Uniform distribution over powers of 2 between min and max inclusive"""
    min_base = np.log2(min)
    max_base = np.log2(max)
    return 2 ** np.random.randint(min_base, max_base + 1)


def uniform(min, max):
    """Uniform distribution between min and max inclusive"""
    return np.random.uniform(min, max)


def int_uniform(min, max):
    """Uniform distribution between min and max inclusive"""
    return np.random.randint(min, max + 1)


def sample_hyperparameters(sweep_config):
    samples = {}
    for name, param in sweep_config.items():
        if name in ("method", "name", "metric"):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            samples[name] = sample_hyperparameters(param)
        elif "values" in param:
            assert "distribution" not in param
            samples[name] = random.choice(param["values"])
        elif "distribution" in param:
            if param["distribution"] == "uniform":
                samples[name] = uniform(param["min"], param["max"])
            elif param["distribution"] == "int_uniform":
                samples[name] = int_uniform(param["min"], param["max"])
            elif param["distribution"] == "uniform_pow2":
                samples[name] = uniform_pow2(param["min"], param["max"])
            elif param["distribution"] == "log_normal":
                samples[name] = log_normal(
                    param["mean"], param["scale"], param["clip"]
                )
            elif param["distribution"] == "logit_normal":
                samples[name] = logit_normal(
                    param["mean"], param["scale"], param["clip"]
                )
            else:
                raise ValueError(
                    f'Invalid distribution: {param["distribution"]}'
                )
        else:
            raise ValueError("Must specify either values or distribution")

    return samples


def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def make_policy(env, config):
    """Create a policy based on the environment."""
    return LateFusionTransformer(
        input_dim=config.train.network.input_dim,
        action_dim=env.single_action_space.n,
        hidden_dim=config.train.network.hidden_dim,
        pred_heads_arch=config.train.network.pred_heads_arch,
        dropout=config.train.network.dropout,
    ).to(config.train.device)


def train(args, vecenv):
    """Main training loop for the PPO agent."""

    policy = make_policy(env=vecenv.driver_env, config=args).to(
        args.train.device
    )

    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    wandb_run = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.update(dict(wandb_run.config.train))

    data = ppo.create(args.train, vecenv, policy, wandb=wandb_run)
    while data.global_step < args.train.total_timesteps:
        try:
            ppo.evaluate(data)  # Rollout
            ppo.train(data)  # Update policy
        except KeyboardInterrupt:
            ppo.close(data)
            os._exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")  # Log the error
            Console().print_exception()
            os._exit(1)  # Exit with a non-zero status to indicate an error

    ppo.evaluate(data)
    ppo.close(data)


def set_experiment_metadata(config):
    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
    if config["train"]["resample_scenes"]:
        if config["train"]["resample_scenes"]:
            dataset_size = config["train"]["resample_dataset_size"]
        config["train"]["exp_id"] = f"PPO_R_{dataset_size}__{datetime_}"
    else:
        dataset_size = str(config["environment"]["k_unique_scenes"])
        config["train"]["exp_id"] = f"PPO_S_{dataset_size}__{datetime_}"

    config["environment"]["dataset_size"] = dataset_size


def init_wandb(args, name, id=None, resume=True, tag=None):
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        mode=args.wandb.mode,
        tags=args.wandb.tags,
        config={
            "environment": dict(args.environment),
            "train": dict(args.train),
            "vec": dict(args.vec),
        },
        name=name,
        save_code=True,
        resume=False,
    )

    return wandb


@app.command()
def run(
    config_path: Annotated[
        str, typer.Argument(help="The path to the default configuration file")
    ] = "examples/experiments/ippo_ff_p1_self_play.yaml",
    *,
    # fmt: off
    # Environment options
    num_worlds: Annotated[Optional[int], typer.Option(help="Number of parallel envs")] = None,
    k_unique_scenes: Annotated[Optional[int], typer.Option(help="The number of unique scenes to sample")] = None,
    dist_to_goal_threshold: Annotated[Optional[float], typer.Option(help="The distance threshold for goal-achieved")] = None,
    sampling_seed: Annotated[Optional[int], typer.Option(help="The seed for sampling scenes")] = None,
    obs_radius: Annotated[Optional[float], typer.Option(help="The radius for the observation")] = None,
    collision_behavior: Annotated[Optional[str], typer.Option(help="The collision behavior; 'ignore' or 'remove'")] = None,
    remove_non_vehicles: Annotated[Optional[int], typer.Option(help="Remove non-vehicles from the scene; 0 or 1")] = None,
    # Train options
    seed: Annotated[Optional[int], typer.Option(help="The seed for training")] = None,
    learning_rate: Annotated[Optional[float], typer.Option(help="The learning rate for training")] = None,
    resample_scenes: Annotated[Optional[int], typer.Option(help="Whether to resample scenes during training; 0 or 1")] = None,
    resample_interval: Annotated[Optional[int], typer.Option(help="The interval for resampling scenes")] = None,
    resample_dataset_size: Annotated[Optional[int], typer.Option(help="The size of the dataset to sample from")] = None,
    total_timesteps: Annotated[Optional[int], typer.Option(help="The total number of training steps")] = None,
    ent_coef: Annotated[Optional[float], typer.Option(help="Entropy coefficient")] = None,
    update_epochs: Annotated[Optional[int], typer.Option(help="The number of epochs for updating the policy")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="The batch size for training")] = None,
    minibatch_size: Annotated[Optional[int], typer.Option(help="The minibatch size for training")] = None,
    gamma: Annotated[Optional[float], typer.Option(help="The discount factor for rewards")] = None,
    collision_weight: Annotated[Optional[float], typer.Option(help="The weight for collision penalty")] = None,
    off_road_weight: Annotated[Optional[float], typer.Option(help="The weight for off-road penalty")] = None,
    goal_achieved_weight: Annotated[Optional[float], typer.Option(help="The weight for goal-achieved reward")] = None,
    collision_penalty_warmup: Annotated[Optional[int], typer.Option(help="Whether to use a warmup period for the collision penalties; 0 or 1")] = None,
    penalty_start_frac: Annotated[Optional[float], typer.Option(help="The fraction of the total timesteps to start applying penalties")] = None,
    warmup_steps: Annotated[Optional[int], typer.Option(help="The number of warmup steps for the penalties")] = None,
    anneal_entropy: Annotated[Optional[int], typer.Option(help="Linearly anneal the entropy coefficient; 0 or 1")] = None,
    anneal_lr: Annotated[Optional[int], typer.Option(help="Linearly anneal the lr rate; 0 or 1")] = None,
    
    # Mode
    mode: Annotated[str, typer.Option(help="The mode to run; 'train' or 'sweep'")] = "train",
    
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
    max_runs: Annotated[Optional[int], typer.Option(help="Maximum number of sweep runs")] = 100,
    render: Annotated[Optional[int], typer.Option(help="Whether to render the environment; 0 or 1")] = None,
):
    """Run PPO training with the given configuration."""
    # fmt: on

    # Load default configs
    config = load_config(config_path)
    
    # Update config.mode if not None
    if mode is not None:
        config.mode = mode

    # Override configs with command-line arguments
    env_config = {
        "num_worlds": num_worlds,
        "k_unique_scenes": k_unique_scenes,
        "dist_to_goal_threshold": dist_to_goal_threshold,
        "sampling_seed": sampling_seed,
        "obs_radius": obs_radius,
        "collision_behavior": collision_behavior,
        "remove_non_vehicles": None
        if remove_non_vehicles is None
        else bool(remove_non_vehicles),
    }
    config.environment.update(
        {k: v for k, v in env_config.items() if v is not None}
    )
    train_config = {
        "seed": seed,
        "learning_rate": learning_rate,
        "resample_scenes": None
        if resample_scenes is None
        else bool(resample_scenes),
        "resample_interval": resample_interval,
        "resample_dataset_size": resample_dataset_size,
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "update_epochs": update_epochs,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "render": None if render is None else bool(render),
        "anneal_entropy": None if anneal_entropy is None else bool(anneal_entropy),
        "anneal_lr": None if anneal_lr is None else bool(anneal_lr),
        "gamma": gamma,
        "collision_weight": collision_weight,
        "off_road_weight": off_road_weight,
        "goal_achieved_weight": goal_achieved_weight,
        "collision_penalty_warmup": None if collision_penalty_warmup is None else bool(collision_penalty_warmup),
        "penalty_start_frac": penalty_start_frac,
        "warmup_steps": warmup_steps,
    }
    config.train.update(
        {k: v for k, v in train_config.items() if v is not None}
    )

    wandb_config = {
        "project": project,
        "entity": entity,
        "group": group,
    }
    config.wandb.update(
        {k: v for k, v in wandb_config.items() if v is not None}
    )

    config["train"]["device"] = config["train"].get(
        "device", "cpu"
    )  # Default to 'cpu' if not set
    if torch.cuda.is_available():
        config["train"]["device"] = "cuda"  # Set to 'cuda' if available

    # Make dataloader
    train_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.environment.num_worlds,
        dataset_size=config.train.resample_dataset_size
        if config.train.resample_scenes
        else config.environment.k_unique_scenes,
        sample_with_replacement=config.train.sample_with_replacement,
        shuffle=config.train.shuffle_dataset,
    )

    # Make environment
    make_env = env_creator(
        data_loader=train_loader,
        environment_config=config.environment,
        train_config=config.train,
        device=config.train.device,
    )
    vecenv = pufferlib.vector.make(
        make_env,
        num_envs=1,  # GPUDrive is already batched
        num_workers=config.vec.num_workers,
        batch_size=config.vec.env_batch_size,
        zero_copy=config.vec.zero_copy,
        backend=pufferlib.vector.Native,
    )

    if config.mode == "train":
        set_experiment_metadata(config)
        train(config, vecenv)
    elif config.mode == "sweep":
        for i in range(max_runs):
            np.random.seed(int(time.time()))
            random.seed(int(time.time()))
            set_experiment_metadata(config)
            hypers = sample_hyperparameters(config.sweep)
            config.train.update(hypers["train"])
            train(config, vecenv)


if __name__ == "__main__":
    app()
