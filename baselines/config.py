
from dataclasses import dataclass
import torch

@dataclass
class ExperimentConfig:
    """
    Configurations for experiments.
    """
    # Rendering options
    render: bool = False
    render_mode: str = "rgb_array"
    render_freq: int = 1
    
    # TODO: Logging
    log_dir: str = "logs"
    
    # Hyperparameters
    policy: str = "MlpPolicy"
    seed: int = 42
    n_steps: int = 2048
    batch_size: int = 256
    verbose: int = 0