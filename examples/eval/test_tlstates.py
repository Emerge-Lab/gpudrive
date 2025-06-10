import numpy as np
import mediapy as media
import wandb

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import TrafficLightObs


if __name__ == "__main__":
    env_config = EnvConfig(
        guidance=True,
        guidance_mode="log_replay",  # Options: "log_replay", "vbd_amortized"
        add_reference_pos_xy=True,
        add_reference_speed=True,
        add_reference_heading=True,
        add_previous_action=True,
        reward_type="guided_autonomy",
        init_mode="wosac_train",
        dynamics_model="delta_local",  # "state", #"classic",
        smoothen_trajectory=True,
        smoothness_weight=0.001,
        collision_weight=-0.01,
        off_road_weight=-0.01,
        guidance_heading_weight=0.01,
        guidance_speed_weight=0.01,
    )

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/tl",
        batch_size=1,
        dataset_size=100,
        sample_with_replacement=False,
        shuffle=False,
        file_prefix="",
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=32,  # Number of agents to control
        device="cpu",
    )

    control_mask = env.cont_agent_mask

    _ = env.reset(mask=control_mask)

    tl_states = env.sim.tl_state_tensor().to_torch()

    print("tl_states.shape", tl_states.shape)
    print("tl_states", tl_states.max())
    
    
    from gpudrive.datatypes.observation import TrafficLightObs
    tl_obs = TrafficLightObs.from_tensor(tl_states_tensor=env.sim.tl_state_tensor())
    print("tl_obs", tl_obs)