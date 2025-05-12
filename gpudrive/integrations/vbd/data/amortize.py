"""
VBD Amortization Script

This script adds pre-computed VBD trajectories to Waymo Open Dataset JSON files.
It uses a sliding window approach to generate trajectories in chunks.

Usage:
    python amortize.py
        --model_path /path/to/vbd/model
        --input_dir /path/to/input/json/files
        --output_dir /path/to/output/json/files
        --batch_size 8
        --window_size 10
        --total_steps 91
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.integrations.vbd.sim_agent.sim_actor import VBDTest
import madrona_gpudrive


def load_vbd_model(model_path, device="cpu", max_cont_agents=64):
    """Load the VBD model from a checkpoint"""
    model = VBDTest.load_from_checkpoint(model_path, torch.device(device))
    model.reset_agent_length(max_cont_agents)
    # Make sure model is in eval mode
    _ = model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="VBD Amortization Script")
    parser.add_argument(
        "--model_path",
        help="Path to the VBD model checkpoint",
        default="gpudrive/integrations/vbd/weights/epoch=18.ckpt",
    )
    parser.add_argument(
        "--input_dir",
        help="Directory containing input JSON files",
        default="data/processed/wosac/validation_json_100",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for output JSON files",
        default="data/processed/wosac/validation_json_100",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for processing"
    )
    parser.add_argument(
        "--window_size", type=int, default=10, help="Size of sliding window"
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=91,
        help="Total number of steps to generate",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=100,
        help="Number of scenes to process",
    )
    args = parser.parse_args()

    # Always use CPU device
    device = "cpu"
    MAX_CONTROLLED_AGENTS = 32
    print(f"Loading VBD model on {device}...")
    vbd_model = load_vbd_model(args.model_path, device, MAX_CONTROLLED_AGENTS)

    # Find all JSON files in the input directory
    json_files = list(Path(args.input_dir).glob("**/*.json"))

    print(f"Found {len(json_files)} JSON files to process")

    # Init GPUDrive env
    INIT_STEPS = 10
    env_config = EnvConfig(
        init_steps=INIT_STEPS,  # Warmup period
        dynamics_model="state",  # Use state-based dynamics model
        dist_to_goal_threshold=1e-5,  # Trick to make sure the agents don't disappear when they reach the goal
        init_mode="wosac_eval",
        max_controlled_agents=MAX_CONTROLLED_AGENTS,
        goal_behavior="ignore",
    )

    # Make env
    gpudrive_env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=SceneDataLoader(
            root=args.input_dir,
            batch_size=args.batch_size,
            dataset_size=args.num_scenes,
            file_prefix="",
        ),
        render_config=RenderConfig(),
        max_cont_agents=MAX_CONTROLLED_AGENTS,  # Maximum number of agents to control per scene
        device=device,
    )

    # Final VBD trajectories to write
    output_trajectories = torch.zeros(
        gpudrive_env.num_worlds,
        gpudrive_env.max_agent_count,
        madrona_gpudrive.kTrajectoryLength,
        6,
    )

    # Save init steps from logs
    log_trajectory = LogTrajectory.from_tensor(
        gpudrive_env.sim.expert_trajectory_tensor(),
        num_worlds=gpudrive_env.num_worlds,
        max_agents=gpudrive_env.max_agent_count,
        device=device,
    )

    output_trajectories[:, :, : INIT_STEPS + 1, :2] = log_trajectory.pos_xy[
        :, :, : INIT_STEPS + 1
    ]
    output_trajectories[:, :, : INIT_STEPS + 1, 2] = log_trajectory.yaw[
        :, :, : INIT_STEPS + 1, 0
    ]
    output_trajectories[:, :, : INIT_STEPS + 1, 3:5] = log_trajectory.vel_xy[
        :, :, : INIT_STEPS + 1
    ]
    output_trajectories[:, :, : INIT_STEPS + 1, 5] = log_trajectory.valids[
        :, :, : INIT_STEPS + 1, 0
    ]

    # Action tensor to step through simulation
    predicted_actions = torch.zeros(
        (
            gpudrive_env.num_worlds,
            gpudrive_env.max_agent_count,
            gpudrive_env.episode_len - INIT_STEPS,
            10,
        )
    )

    # World means for VBD outputs
    world_means = (
        gpudrive_env.sim.world_means_tensor().to_torch()[:, :2].to(device)
    )

    for _ in range(int(args.num_scenes / args.batch_size)):
        # Generate VBD input
        scene_context = gpudrive_env.construct_context(
            init_steps=gpudrive_env.init_steps
        )
        # Controlled agent mask
        world_agent_indices = scene_context["agents_id"]

        # Generate VBD output
        predictions = vbd_model.sample_denoiser(scene_context)
        vbd_output = predictions["denoised_trajs"].to(device).detach()

        for i in range(gpudrive_env.num_worlds):
            # Get controlled agent indices for this world
            valid_mask = (
                world_agent_indices[i] >= 0
            )  # Boolean mask of valid indices
            valid_world_indices = world_agent_indices[i][
                valid_mask
            ]  # Filtered tensor

            # Populate predicted actions
            predicted_actions[i, valid_world_indices, :, :2] = vbd_output[
                i, valid_world_indices, :, :2
            ] - world_means[i].view(1, 1, 2)
            predicted_actions[i, valid_world_indices, :, 3] = vbd_output[
                i, valid_world_indices, :, 2
            ]
            predicted_actions[i, valid_world_indices, :, 4:6] = vbd_output[
                i, valid_world_indices, :, 3:5
            ]

            # Populate output trajectories
            output_trajectories[
                i, valid_world_indices, INIT_STEPS + 1 :, :2
            ] = vbd_output[i, valid_world_indices, :, :2] - world_means[
                i
            ].view(
                1, 1, 2
            )
            output_trajectories[
                i, valid_world_indices, INIT_STEPS + 1 :, 2:5
            ] = vbd_output[i, valid_world_indices, :, 2:]
            output_trajectories[
                i, valid_world_indices, INIT_STEPS + 1 :, 5
            ] = 1.0

        # Save to each file's json
        index_to_id = GlobalEgoState.from_tensor(
            gpudrive_env.sim.absolute_self_observation_tensor(), device=device
        ).id

        filenames = gpudrive_env.get_scenario_ids()

        for i in range(gpudrive_env.num_worlds):
            filename = f"{filenames[i]}.json"
            with open(f"{args.input_dir}/{filename}", "r") as f:
                data = json.load(f)

            valid_mask = (
                world_agent_indices[i] >= 0
            )  # Boolean mask of valid indices
            valid_world_indices = world_agent_indices[i][
                valid_mask
            ]  # Filtered tensor

            # Find object with correct id
            for j in valid_world_indices:
                id = index_to_id[i, j]

                for obj in data["objects"]:
                    if obj["id"] == id:
                        # Add VBD output to JSON
                        obj["vbd_trajectory"] = (
                            output_trajectories[i, j].cpu().numpy().tolist()
                        )
                        break

            # Write to output directory
            output_file = os.path.join(args.output_dir, filename)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)

        # Load next batch of scenes
        try:
            gpudrive_env.swap_data_batch()
        except Exception as e:
            print(f"Reached end of dataset: {e}")
            break

    print("VBD amortization complete!")


if __name__ == "__main__":
    main()

    """
    python gpudrive/integrations/vbd/data/amortize.py --input_dir data/processed/wosac/validation_json_100 --output_dir data/processed/wosac/validation_json_100 --batch_size 8 --window_size 10 --total_steps 91
    """
