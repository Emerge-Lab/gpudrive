from pdb import set_trace as T
import argparse
from datetime import datetime
import os
import wandb
import torch

import pufferlib
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console

from integrations.rl.puffer import ppo
from integrations.rl.puffer.puffer_env import env_creator
from integrations.rl.puffer.utils import Policy


def make_policy(env):
    """Make the policy for the environment"""
    policy = Policy(env)
    return pufferlib.frameworks.cleanrl.Policy(policy)


def train(args):
    args.wandb = None
    if args.track:
        args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
        args.train.__dict__.update(dict(args.wandb.config.train))
    if args.vec.backend == "native":
        backend = pufferlib.vector.Native
    elif args.vec.backend == "serial":
        backend = pufferlib.vector.Serial
    elif args.vec.backend == "multiprocessing":
        backend = pufferlib.vector.Multiprocessing
    elif args.vec == "ray":
        backend = pufferlib.vector.Ray
    else:
        raise ValueError(
            f"Invalid --vec.backend (native/serial/multiprocessing/ray)."
        )

    # Make vectorized environment
    vecenv = pufferlib.vector.make(
        make_env,
        num_envs=args.vec.num_envs,
        num_workers=args.vec.num_workers,
        batch_size=args.vec.env_batch_size,
        zero_copy=args.vec.zero_copy,
        backend=backend,
    )

    # Make policy
    policy = make_policy(vecenv.driver_env).to(args.train.device)

    args.train.env = args.env

    # Training loop
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
    """Initialize WandB."""

    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            "train": dict(args.train),
            "vec": dict(args.vec),
        },
        name=name,
        save_code=True,
        resume=resume,
    )
    return wandb


def sweep(args, project="PPO"):
    """Initialize a WandB sweep with hyperparameters."""
    sweep_id = wandb.sweep(
        sweep=dict(
            method="random",
            name=sweep,
            metric=dict(
                goal="maximize",
                name="environment/episode_return",
            ),
            parameters=dict(
                learning_rate=dict(
                    distribution="log_uniform_values", min=1e-4, max=1e-1
                ),
                batch_size=dict(
                    values=[512, 1024, 2048],
                ),
                minibatch_size=dict(
                    values=[128, 256, 512],
                ),
            ),
        ),
        project=project,
    )

    args.track = True
    wandb.agent(sweep_id, lambda: train(args), count=100)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib PPO [bright_cyan]{pufferlib.__version__}[/]"
        "arguments. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--env", type=str, default="gpudrive")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed/examples"
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=50,
        help="Number of parallel worlds (environments)",
    )
    parser.add_argument(
        "--k-unique-scenes",
        type=int,
        default=1,
        help="Number of unique scenes to train on",
    )
    parser.add_argument(
        "--max-cont-agents",
        type=int,
        default=32,
        help="Number of agents per scenario",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices="train eval evaluate sweep autotune baseline profile".split(),
    )

    parser.add_argument("--use-rnn", action="store_true")
    parser.add_argument(
        "--eval-model-path",
        type=str,
        default=None,
        help="Path to model to evaluate",
    )
    parser.add_argument("--baseline", action="store_true", help="Baseline run")

    parser.add_argument(
        "--wandb-entity", type=str, default="", help="WandB entity"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="pufferlib-integration",
        help="WandB project",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="my_project",
        help="WandB group",
    )
    parser.add_argument("--track", action="store_true", help="Track on WandB")

    # Train configuration
    parser.add_argument(
        "--train.exp-id",
        type=str,
        default=datetime.now().strftime("%m_%d_%H_%M_%S"),
    )
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument("--train.torch-deterministic", action="store_true")
    parser.add_argument("--train.cpu-offload", action="store_true")
    parser.add_argument(
        "--train.device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--train.total-timesteps", type=int, default=20_000_000
    )
    parser.add_argument("--train.learning-rate", type=float, default=3e-4)
    parser.add_argument("--train.anneal-lr", action="store_false")
    parser.add_argument("--train.gamma", type=float, default=0.99)
    parser.add_argument("--train.gae-lambda", type=float, default=0.95)
    parser.add_argument("--train.update-epochs", type=int, default=5)
    parser.add_argument("--train.norm-adv", action="store_true")
    parser.add_argument("--train.clip-coef", type=float, default=0.2)
    parser.add_argument(
        "--train.clip-vloss", type=bool, default=False
    )  # No clipping on the VF by default
    parser.add_argument("--train.vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--train.ent-coef", type=float, default=0.0001)
    parser.add_argument("--train.vf-coef", type=float, default=0.5)
    parser.add_argument("--train.max-grad-norm", type=float, default=0.5)
    parser.add_argument("--train.target-kl", type=float, default=None)
    parser.add_argument("--train.checkpoint-interval", type=int, default=5000)
    parser.add_argument("--train.checkpoint-path", type=str, default="./runs")
    parser.add_argument("--train.render", type=bool, default=True)
    parser.add_argument(
        "--train.render-interval",
        type=int,
        default=1000,
        help="Frequency to render the environment in epochs",
    )
    parser.add_argument(
        "--train.batch-size", type=int, default=25_000
    )  # Number of steps per rollout
    parser.add_argument("--train.minibatch-size", type=int, default=5_000)
    parser.add_argument(
        "--train.bptt-horizon", type=int, default=50
    )  # Not used
    parser.add_argument("--train.compile", action="store_true")
    parser.add_argument(
        "--train.compile-mode", type=str, default="reduce-overhead"
    )
    parser.add_argument(
        "--train.animate-learning",
        default=False,
        help="Animate the learning process.",
    )
    parser.add_argument(
        "--vec.backend",
        type=str,
        default="native",
        choices="serial multiprocessing ray native".split(),
    )
    parser.add_argument("--vec.num-envs", type=int, default=1)
    parser.add_argument("--vec.num-workers", type=int, default=1)
    parser.add_argument("--vec.env-batch-size", type=int, default=1)
    parser.add_argument("--vec.zero-copy", action="store_true")
    parsed = parser.parse_args()

    args = {}
    for k, v in vars(parsed).items():
        if "." in k:
            group, name = k.split(".")
            if group not in args:
                args[group] = {}

            args[group][name] = v
        else:
            args[k] = v

    args["train"] = pufferlib.namespace(**args["train"])
    args["vec"] = pufferlib.namespace(**args["vec"])

    args = pufferlib.namespace(**args)

    make_env = env_creator(
        name=args.env,
        data_dir=args.data_dir,
        num_worlds=args.num_worlds,
        max_cont_agents=args.max_cont_agents,
        k_unique_scenes=args.k_unique_scenes,
    )

    if args.mode == "train":
        train(args)
    elif args.mode in ("eval", "evaluate"):
        try:
            ppo.rollout(
                make_env,
                env_kwargs={},
                agent_creator=make_policy,
                model_path=args.eval_model_path,
                device=args.train.device,
            )
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == "sweep":
        sweep(args)
    elif args.mode == "autotune":
        pufferlib.vector.autotune(make_env, batch_size=args.vec.env_batch_size)
