from dataclasses import dataclass


@dataclass
class BehavCloningConfig:

    # Dataset & device
    data_dir: str = "example_data"
    device: str = "cuda"

    # Number of scenarios / worlds
    num_worlds: int = 3
    max_cont_agents: int = 128

    # Discretize actions and use action indices
    discretize_actions: bool = True
    use_action_indices: bool = True
    # Record a set of trajectories as sanity check
    make_sanity_check_video: bool = True

    # Logging
    wandb_mode: str = "online"

    # Hyperparameters
    batch_size: int = 512
    epochs: int = 2000
    lr: float = 1e-3
    hidden_size: int = 256
