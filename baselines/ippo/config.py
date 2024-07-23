from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configurations for experiments."""

    # DATASET & DEVICE
    data_dir: str = "maps_10"
    train_on_k_unique_scenes = 10  # If set to a non-zero value K, randomly samples K unique scenarios for a total of num_worlds scenes

    # NUM ENVIRONMENTS
    num_worlds: int = 50
    device: str = "cuda"

    # RENDERING
    render: bool = True
    render_mode: str = "rgb_array"
    render_freq: int = 200  # Render every k rollouts
    render_n_worlds: int = 10  # Number of worlds to render

    track_time_to_solve: bool = True

    # LOGGING & WANDB
    sync_tensorboard: bool = True
    logging_collection_window: int = (
        100  # How many trajectories we average logs over
    )
    log_freq: int = 100
    project_name = "gpudrive_benchmark"
    group_name = "verify"
    entity = "_emerge"
    tags = ["IPPO", "LATE_FUSION", "PERM_EQ"]
    wandb_mode = "online"  # Options: online, offline, disabled

    # MODEL CHECKPOINTING
    save_policy: bool = True
    save_policy_freq: int = 500

    # HYPERPARAMETERS
    seed: int = 42
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    n_steps: int = 92  # Has to be at least > episode_length = 91
    num_minibatches: int = 5  # Used to determine the minibatch size
    verbose: int = 0
    total_timesteps: int = 6e7
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    lr: float = 3e-4
    n_epochs: int = 5

    # NETWORK
    mlp_class = LateFusionNet
    policy = LateFusionPolicy
    ego_state_layers = [64, 32]
    road_object_layers = [64, 64]
    road_graph_layers = [64, 64]
    shared_layers = [64, 64]
    act_func = "tanh"
    dropout = 0.0
    last_layer_dim_pi = 64
    last_layer_dim_vf = 64
