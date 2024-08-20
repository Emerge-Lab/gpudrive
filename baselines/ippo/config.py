from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy
from dataclasses import dataclass
from pygpudrive.env.config import SelectionDiscipline
@dataclass
class ExperimentConfig:
    """Configurations for experiments."""

    # DATASET
    data_dir: str = "data"

    # NUM PARALLEL ENVIRONMENTS & DEVICE
    num_worlds: int = 50 # Number of parallel environments
    # How to select scenes from the dataset
    selection_discipline = SelectionDiscipline.PAD_N # K_UNIQUE_N / PAD_N
    k_unique_scenes: int = None
    device: str = "cuda"  # or "cpu"

    # RENDERING
    render: bool = True
    render_mode: str = "rgb_array"
    render_freq: int = 50  # Render every k rollouts
    render_n_worlds: int = 3  # Number of worlds to render

    # TRACK THE TIME IT TAKES TO GET TO 95% GOAL RATE
    track_time_to_solve: bool = False

    # LOGGING & WANDB
    sync_tensorboard: bool = True
    logging_collection_window: int = (
        100  # How many trajectories we average logs over
    )
    log_freq: int = 100
    project_name = "my_gpudrive_tests"
    group_name = " "
    entity = " "
    tags = ["IPPO", "LATE_FUSION", "PERM_EQ"]
    wandb_mode = "online"  # Options: online, offline, disabled

    # MODEL CHECKPOINTING
    save_policy: bool = True
    save_policy_freq: int = 100

    # HYPERPARAMETERS
    seed: int = 42
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    n_steps: int = 91
    num_minibatches: int = 5  # Used to determine the minibatch size
    verbose: int = 0
    total_timesteps: int = 1e7
    ent_coef: float = 0.00
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
