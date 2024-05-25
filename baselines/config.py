from networks.basic_ffn import FeedForwardPolicy
from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    """Configurations for experiments."""

    # General
    device: str = "cuda"

    # Rendering options
    render: bool = False
    render_mode: str = "rgb_array"
    render_freq: int = 10

    # TODO: Logging
    log_dir: str = "logs"
    use_wandb: bool = True

    # Hyperparameters
    policy: torch.nn.Module = FeedForwardPolicy
    seed: int = 42
    n_steps: int = 180
    batch_size: int = 180
    verbose: int = 0
    total_timesteps: int = 50_000_000
