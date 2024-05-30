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
    render_freq: int = 10000  # Render after every kth rollout
    render_n_worlds: int = 1

    # TODO: Logging
    log_dir: str = "logs"
    use_wandb: bool = True
    logging_collection_window: int = (
        1000  # how many trajectories we average logs over
    )

    # Hyperparameters
    policy: torch.nn.Module = FeedForwardPolicy
    seed: int = 42
    n_steps: int = 92  # Has to be at least > episode_length = 91
    batch_size: int = 256
    verbose: int = 0
    total_timesteps: int = 150_000_000

    # Wandb
    project_name = "gpudrive"
    group_name = "dc/05/30"
    entity = "_emerge"
