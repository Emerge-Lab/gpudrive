from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy
from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    """Configurations for experiments."""

    # DATASET & DEVICE
    data_dir: str = "formatted_json_v2_no_tl_train"
    device: str = "cuda"

    # RENDERING
    render: bool = True
    render_mode: str = "rgb_array"
    render_freq: int = 100
    # Start rendering success/failure modes after this many global timesteps
    log_failure_modes_after: int = 1e5  # Set to None to disable
    log_success_modes_after: int = 1e5  # Set to None to disable
    render_n_worlds: int = 5  # Number of worlds to render

    # LOGGING & WANDB
    use_wandb: bool = True
    sync_tensorboard: bool = True
    logging_collection_window: int = (
        100  # how many trajectories we average logs over
    )
    log_freq: int = 100
    project_name = "gpudrive"
    group_name = "dc/PPO_cluster"
    entity = "_emerge"
    tags = ["IPPO", "LATE_FUSION", "PERM_EQ"]
    wandb_mode = "online"

    # MODEL CHECKPOINTING
    save_policy: bool = True
    save_policy_freq: int = 1000

    # HYPERPARAMETERS
    seed: int = 42
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    n_steps: int = 92  # Has to be at least > episode_length = 91
    batch_size: int = 2048
    verbose: int = 0
    total_timesteps: int = 3e8
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    lr: float = 3e-4
    n_epochs: int = 10

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
