"""
This implementation is adapted from the demo in PufferLib by Joseph Suarez.
The original code can be found at: https://github.com/PufferAI/PufferLib/blob/dev/demo.py.
"""

import os
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


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    config["train"]["exp_id"] = config["train"][
        "exp_id"
    ] or datetime.now().strftime("%m_%d_%H_%M_%S")
    config["train"]["device"] = (
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    return pufferlib.namespace(**config)


def make_policy(env):
    return pufferlib.frameworks.cleanrl.Policy(Policy(env))


def train(args):
    """Main training loop for the PPO agent."""
    if args.wandb.track:
        args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
        args.train.__dict__.update(dict(args.wandb.config.train))

    backend_mapping = {
        "native": pufferlib.vector.Native,
        "serial": pufferlib.vector.Serial,  # Only native backend is supported with GPUDrive
        "multiprocessing": pufferlib.vector.Multiprocessing,
        "ray": pufferlib.vector.Ray,
    }
    backend = backend_mapping.get(args.vec.backend)
    if not backend:
        raise ValueError("Invalid --vec.backend.")

    vecenv = pufferlib.vector.make(
        make_env,
        num_envs=args.vec.num_envs,
        num_workers=args.vec.num_workers,
        batch_size=args.vec.env_batch_size,
        zero_copy=args.vec.zero_copy,
        backend=backend,
    )

    policy = make_policy(vecenv.driver_env).to(args.train.device)
    args.train.env = args.env

    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
    while data.global_step < args.train.total_timesteps:
        try:
            ppo.evaluate(data)  # Rollout
            ppo.train(data)  # Update policy
        except KeyboardInterrupt:
            ppo.close(data)
            os._exit(0)
        except Exception:
            Console().print_exception()
            os._exit(0)

    ppo.evaluate(data)
    ppo.close(data)


def init_wandb(args, name, id=None, resume=True):
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        config={"train": dict(args.train), "vec": dict(args.vec)},
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

    args.track = True
    wandb.agent(sweep_id, lambda: train(args), count=100)


if __name__ == "__main__":

    config = load_config("baselines/ippo/config/ippo_ff_single_scene.yaml")

    make_env = env_creator(
        name=config.env,
        data_dir=config.data_dir,
        num_worlds=config.num_worlds,
        max_cont_agents=config.max_cont_agents,
        k_unique_scenes=config.k_unique_scenes,
    )

    if config.mode == "train":
        train(config)
