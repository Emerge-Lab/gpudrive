import importlib
import gpudrive

importlib.reload(gpudrive)
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
import torch
import numpy as np


if __name__ == "__main__":

    GUIDANCE_MODE = "vbd_online"
    DATASET = "data/processed/wosac/validation_json_100"  # Ensure VBD trajectory structures are in here
    SAVE_PATH = "examples/eval/figures_data/guidance/"

    env_config = EnvConfig(
        dynamics_model="classic",
        reward_type="guided_autonomy",
        guidance=True,
        guidance_mode=GUIDANCE_MODE,
        add_reference_heading=True,
        add_reference_speed=True,
        add_reference_pos_xy=True,
        init_mode="wosac_eval",
        smoothen_trajectory=False,
    )
    render_config = RenderConfig()

    train_loader = SceneDataLoader(
        root=DATASET,
        batch_size=10,
        dataset_size=100,
        sample_with_replacement=False,
        shuffle=False,
        file_prefix="",
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=32,
        device="cuda",
    )

    obs = env.reset(env.cont_agent_mask)

    # Save for analysis
    reference_traj = torch.cat(
        [
            env.reference_trajectory.pos_xy,
            env.reference_trajectory.vel_xy,
            env.reference_trajectory.yaw,
            env.reference_trajectory.valids,
        ],
        dim=-1,
    )
    reference_traj_np = (
        reference_traj[env.cont_agent_mask].detach().cpu().numpy()
    )

    np.save(
        f"{SAVE_PATH}reference_{GUIDANCE_MODE}.npy", reference_traj_np
    )
    print(f"Saved reference trajectory for {GUIDANCE_MODE} mode.")
