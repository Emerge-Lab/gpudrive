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
import wandb
from box import Box
from integrations.rl.puffer import ppo
from integrations.rl.puffer.puffer_env import env_creator
from integrations.rl.puffer.utils import Policy

import pufferlib
import pufferlib.vector
import pufferlib.frameworks.cleanrl
from rich.console import Console

import typer
from typer import Typer

app = Typer()


def load_config(config_path):
    """Load the configuration file."""
    # fmt: off
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))

    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S")

    if config["train"]["resample_scenes"]:

        if config["train"]["resample_scenes"]:
            dataset_size = config["train"]["resample_dataset_size"]

        config["train"]["exp_id"] = f'{config["train"]["exp_id"]}__R_{dataset_size}__{datetime_}'
    else:
        dataset_size = str(config["environment"]["k_unique_scenes"])
        config["train"]["exp_id"] = f'{config["train"]["exp_id"]}__S_{dataset_size}__{datetime_}'

    config["environment"]["dataset_size"] = dataset_size
    config["train"]["device"] = config["train"].get("device", "cpu")  # Default to 'cpu' if not set
    if torch.cuda.is_available():
        config["train"]["device"] = "cuda"  # Set to 'cuda' if available
    # fmt: on
    return pufferlib.namespace(**config)


def make_policy(env):
    """Create a policy based on the environment."""
    return pufferlib.frameworks.cleanrl.Policy(Policy(env))


def train(args, make_env):
    """Main training loop for the PPO agent."""
    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

    backend_mapping = {
        # Note: Only native backend is currently supported with GPUDrive
        "native": pufferlib.vector.Native,
        "serial": pufferlib.vector.Serial,
        "multiprocessing": pufferlib.vector.Multiprocessing,
        "ray": pufferlib.vector.Ray,
    }

    backend = backend_mapping.get(args.vec.backend)
    if not backend:
        raise ValueError("Invalid --vec.backend.")

    vecenv = pufferlib.vector.make(
        make_env,
        num_envs=1,  # GPUDrive is already batched
        num_workers=args.vec.num_workers,
        batch_size=args.vec.env_batch_size,
        zero_copy=args.vec.zero_copy,
        backend=backend,
    )

    policy = make_policy(vecenv.driver_env).to(args.train.device)

    args.train.env = args.environment.name

    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
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


def init_wandb(args, name, id=None, resume=True):
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
        resume=resume,
    )
    return wandb


def sweep(args, project="PPO", sweep_name="my_sweep"):
    """Initialize a WandB sweep with hyperparameters."""
    sweep_id = wandb.sweep(
        sweep=dict(
            method="random",
            name=sweep_name,
            metric={"goal": "maximize", "name": "environment/episode_return"},
            parameters={
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-4,
                    "max": 1e-1,
                },
                "batch_size": {"values": [512, 1024, 2048]},
                "minibatch_size": {"values": [128, 256, 512]},
            },
        ),
        project=project,
    )
    wandb.agent(sweep_id, lambda: train(args), count=100)


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
    collision_weight: Annotated[Optional[float], typer.Option(help="The weight for collision penalty")] = None,
    off_road_weight: Annotated[Optional[float], typer.Option(help="The weight for off-road penalty")] = None,
    goal_achieved_weight: Annotated[Optional[float], typer.Option(help="The weight for goal-achieved reward")] = None,
    dist_to_goal_threshold: Annotated[Optional[float], typer.Option(help="The distance threshold for goal-achieved")] = None,
    sampling_seed: Annotated[Optional[int], typer.Option(help="The seed for sampling scenes")] = None,
    obs_radius: Annotated[Optional[float], typer.Option(help="The radius for the observation")] = None,
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
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
):
    """Run PPO training with the given configuration."""
    # fmt: on

    # Load default configs
    config = load_config(config_path)

    # Override configs with command-line arguments
    env_config = {
        "num_worlds": num_worlds,
        "k_unique_scenes": k_unique_scenes,
        "collision_weight": collision_weight,
        "off_road_weight": off_road_weight,
        "goal_achieved_weight": goal_achieved_weight,
        "dist_to_goal_threshold": dist_to_goal_threshold,
        "sampling_seed": sampling_seed,
        "obs_radius": obs_radius,
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

    make_env = env_creator(
        data_dir=config.data_dir,
        environment_config=config.environment,
        train_config=config.train,
        device=config.train.device,
    )

    if config.mode == "train":
        train(config, make_env)


if __name__ == "__main__":
    app()
