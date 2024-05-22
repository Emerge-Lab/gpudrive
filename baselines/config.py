from networks.basic_ffn import FeedForwardPolicy
from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    """Configurations for experiments."""

    # General
    device: str = "cuda"

    # Dataset
    data_dir: str = "waymo_data"

    # Rendering settings
    render: bool = False
    render_mode: str = "rgb_array"
    render_freq: int = 10
    render_n_worlds: int = 1

    # Hyperparameters
    policy: torch.nn.Module = FeedForwardPolicy
    seed: int = 42
    n_steps: int = 900
    batch_size: int = 512
    verbose: int = 0
    total_timesteps: int = 150_000_000

    # Wandb
    project_name = "gpudrive"
    group_name = "PPO_0521"
    entity = "_emerge"
