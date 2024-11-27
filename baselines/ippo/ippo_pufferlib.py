"""
This implementation is adapted from the demo in PufferLib by Joseph Suarez,
which in turn is adapted from Costa Huang's CleanRL PPO + LSTM implementation.
Links
- PufferLib: https://github.com/PufferAI/PufferLib/blob/dev/demo.py
- Cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import os
import yaml
from datetime import datetime
import torch
import wandb
from box import Box

from integrations.rl.puffer import ppo
from integrations.rl.puffer.puffer_env import env_creator
from integrations.rl.puffer.utils import Policy, LiDARPolicy

import pufferlib
import pufferlib.vector
import pufferlib.frameworks.cleanrl
from rich.console import Console


def load_config(config_path):
    """Load the configuration file."""
    # fmt: off
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))

    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S")
    config["train"]["exp_id"] = f'{config["train"]["exp_id"]}__S_{str(config["environment"]["k_unique_scenes"])}__{datetime_}'
    config["train"]["device"] = config["train"].get("device", "cpu")  # Default to 'cpu' if not set
    if torch.cuda.is_available():
        config["train"]["device"] = "cuda"  # Set to 'cuda' if available
    # fmt: on
    return pufferlib.namespace(**config)


def make_policy(env):
    """Create a policy based on the environment."""
    return pufferlib.frameworks.cleanrl.Policy(Policy(env))


def train(args):
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
            "network": dict(args.network)
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


if __name__ == "__main__":

    config = load_config("baselines/ippo/config/ippo_ff_puffer.yaml")

    make_env = env_creator(
        data_dir=config.data_dir,
        environment_config=config.environment,
        device=config.train.device,
    )

    if config.mode == "train":
        train(config)
