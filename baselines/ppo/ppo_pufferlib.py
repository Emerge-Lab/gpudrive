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

from gpudrive.integrations.puffer import ppo
from gpudrive.env.env_puffer import PufferGPUDrive

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.dataset import SceneDataLoader

import pufferlib
import pufferlib.vector
import pufferlib.cleanrl
from rich.console import Console

import typer
from typer import Typer

app = Typer()


def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def make_agent(env, config):
    """Create a policy based on the environment."""

    if config.continue_training:
        print("Loading checkpoint...")
        # Load checkpoint
        saved_cpt = torch.load(
            f=config.model_cpt,
            map_location=config.train.device,
            weights_only=False,
        )
        policy = NeuralNet(
            input_dim=saved_cpt["model_arch"]["input_dim"],
            action_dim=saved_cpt["action_dim"],
            hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
            config=config.environment,
        )

        # Load the model parameters
        policy.load_state_dict(saved_cpt["parameters"])

        return policy

    else:
        # Start from scratch
        return NeuralNet(
            input_dim=config.train.network.input_dim,
            action_dim=env.single_action_space.n,
            hidden_dim=config.train.network.hidden_dim,
            dropout=config.train.network.dropout,
            config=config.environment,
        )


def train(args, vecenv):
    """Main training loop for the PPO agent."""
    policy = make_agent(env=vecenv.driver_env, config=args).to(
        args.train.device
    )

    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

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
        resume=False,
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
    ] = "baselines/ppo/config/ppo_base_puffer.yaml",
    *,
    # fmt: off
    # Environment options
    num_worlds: Annotated[Optional[int], typer.Option(help="Number of parallel envs")] = None,
    max_controlled_agents: Annotated[Optional[int], typer.Option(help="Number of controlled agents")] = None,
    k_unique_scenes: Annotated[Optional[int], typer.Option(help="The number of unique scenes to sample")] = None,
    collision_weight: Annotated[Optional[float], typer.Option(help="The weight for collision penalty")] = None,
    off_road_weight: Annotated[Optional[float], typer.Option(help="The weight for off-road penalty")] = None,
    goal_achieved_weight: Annotated[Optional[float], typer.Option(help="The weight for goal-achieved reward")] = None,
    dist_to_goal_threshold: Annotated[Optional[float], typer.Option(help="The distance threshold for goal-achieved")] = None,
    randomize_rewards: Annotated[Optional[int], typer.Option(help="If reward_type == reward_conditioned, choose the condition_mode; 0 or 1")] = 0,
    sampling_seed: Annotated[Optional[int], typer.Option(help="The seed for sampling scenes")] = None,
    obs_radius: Annotated[Optional[float], typer.Option(help="The radius for the observation")] = None,
    collision_behavior: Annotated[Optional[str], typer.Option(help="The collision behavior; 'ignore' or 'remove'")] = None,
    remove_non_vehicles: Annotated[Optional[int], typer.Option(help="Remove non-vehicles from the scene; 0 or 1")] = None,
    use_vbd: Annotated[Optional[bool], typer.Option(help="Use VBD model for trajectory predictions")] = False,
    vbd_trajectory_weight: Annotated[Optional[float], typer.Option(help="Weight for VBD trajectory deviation penalty")] = 0.1,
    vbd_in_obs: Annotated[Optional[bool], typer.Option(help="Include VBD predictions in the observation")] = False,
    init_steps: Annotated[Optional[int], typer.Option(help="Environment warmup steps")] = 0,
    # Train options
    seed: Annotated[Optional[int], typer.Option(help="The seed for training")] = None,
    learning_rate: Annotated[Optional[float], typer.Option(help="The learning rate for training")] = None,
    anneal_lr: Annotated[Optional[int], typer.Option(help="Whether to anneal the learning rate over time; 0 or 1")] = None,
    resample_scenes: Annotated[Optional[int], typer.Option(help="Whether to resample scenes during training; 0 or 1")] = None,
    resample_interval: Annotated[Optional[int], typer.Option(help="The interval for resampling scenes")] = None,
    resample_dataset_size: Annotated[Optional[int], typer.Option(help="The size of the dataset to sample from")] = None,
    total_timesteps: Annotated[Optional[int], typer.Option(help="The total number of training steps")] = None,
    ent_coef: Annotated[Optional[float], typer.Option(help="Entropy coefficient")] = None,
    update_epochs: Annotated[Optional[int], typer.Option(help="The number of epochs for updating the policy")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="The batch size for training")] = None,
    minibatch_size: Annotated[Optional[int], typer.Option(help="The minibatch size for training")] = None,
    gamma: Annotated[Optional[float], typer.Option(help="The discount factor for rewards")] = None,
    vf_coef: Annotated[Optional[float], typer.Option(help="Weight for vf_loss")] = None,
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
    render: Annotated[Optional[int], typer.Option(help="Whether to render the environment; 0 or 1")] = None,
):
    """Run PPO training with the given configuration."""
    # fmt: on

    # Load default configs
    config = load_config(config_path)

    if config.environment.reward_type == "reward_conditioned":
        if bool(randomize_rewards):
            config.environment.condition_mode = "random"
            config.train.exp_id = "random_weights"
        else:
            config.environment.condition_mode = (
                "fixed"  # Use the same type for every agent
            )
            config.train.exp_id = "fixed_weights"

    # Override configs with command-line arguments
    env_config = {
        "num_worlds": num_worlds,
        "max_controlled_agents": max_controlled_agents,
        "k_unique_scenes": k_unique_scenes,
        "collision_weight": collision_weight,
        "off_road_weight": off_road_weight,
        "goal_achieved_weight": goal_achieved_weight,
        "dist_to_goal_threshold": dist_to_goal_threshold,
        "sampling_seed": sampling_seed,
        "obs_radius": obs_radius,
        "collision_behavior": collision_behavior,
        "remove_non_vehicles": None
        if remove_non_vehicles is None
        else bool(remove_non_vehicles),
        "use_vbd": use_vbd,
        "vbd_trajectory_weight": vbd_trajectory_weight,
        "vbd_in_obs": vbd_in_obs,
        "init_steps": init_steps,
    }
    config.environment.update(
        {k: v for k, v in env_config.items() if v is not None}
    )

    train_config = {
        "seed": seed,
        "learning_rate": learning_rate,
        "anneal_lr": None if anneal_lr is None else bool(anneal_lr),
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
        "gamma": gamma,
        "vf_coef": vf_coef,
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

    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]

    if config["continue_training"]:
        cont_train = "C"
    else:
        cont_train = ""

    if config["train"]["resample_scenes"]:
        if config["train"]["resample_scenes"]:
            dataset_size = config["train"]["resample_dataset_size"]
        config["train"][
            "exp_id"
        ] = f'{config["train"]["exp_id"]}__{cont_train}__R_{dataset_size}__{datetime_}'
    else:
        dataset_size = str(config["environment"]["k_unique_scenes"])
        config["train"][
            "exp_id"
        ] = f'{config["train"]["exp_id"]}__{cont_train}__S_{dataset_size}__{datetime_}'

    config["environment"]["dataset_size"] = dataset_size
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
    vecenv = PufferGPUDrive(
        data_loader=train_loader,
        **config.environment,
        **config.train,
    )

    train(config, vecenv)


if __name__ == "__main__":

    app()
