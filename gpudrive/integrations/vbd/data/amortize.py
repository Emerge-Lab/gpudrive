"""
VBD Amortization Script (CPU-only version)

This script adds pre-computed VBD trajectories to Waymo Open Dataset JSON files.
It uses a sliding window approach to generate trajectories in chunks, running exclusively on CPU.

Usage:
    python vbd_amortize_cpu.py --model_path /path/to/vbd/model 
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
from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.integrations.vbd.sim_agent.sim_actor import VBDTest

def load_vbd_model(model_path, device="cpu"):
    """Load the VBD model from a checkpoint, forcing CPU usage."""
    # Force the model to be loaded on CPU
    model = VBDTest.load_from_checkpoint(model_path, torch.device(device))
    # Make sure model is in eval mode
    _ = model.eval()
    return model

def process_file(json_path, vbd_model, output_dir, batch_size=1, window_size=10, total_steps=91):
    """Process a single JSON file, adding VBD trajectories, using CPU only."""
    print(f"Processing file: {json_path}")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create config
    config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        bev_obs=False,
        lidar_obs=False,
        disable_classic_obs=False,
        max_controlled_agents=32,
        use_vbd=False,  # We don't need the VBD integration in the env
        dynamics_model="delta_local",
    )
    
    # Create data loader with just this file
    data_loader = SceneDataLoader(
        root=os.path.dirname(json_path),
        batch_size=batch_size,
        dataset_size=1,
        sample_with_replacement=False,
        shuffle=False,
    )
    
    # Create environment with CPU device
    env = GPUDriveTorchEnv(
        config=config,
        data_loader=data_loader,
        max_cont_agents=64,
        device="cpu",  # Force CPU
        action_type="discrete",
        backend="torch",
    )
    
    # Reset the environment
    _ = env.reset()
    
    # Get initial world state
    means_xy = env.sim.world_means_tensor().to_torch().to("cpu")[:, :2]
    
    # Initialize VBD trajectories dictionary
    vbd_trajectories = {}
    for i, obj in enumerate(data["objects"]):
        agent_id = obj["id"]
        vbd_trajectories[agent_id] = [None] * total_steps
    
    # Process in sliding windows
    for start_idx in range(0, total_steps, window_size):
        print(f"Processing window starting at {start_idx}")
        
        # Generate sample batch for VBD from current state, ensuring it's on CPU
        sample_batch = env._generate_sample_batch(init_steps=window_size)
        
        # Make sure all tensors in the batch are on CPU
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                sample_batch[key] = value.to("cpu")
        
        # Generate predictions using CPU
        predictions = vbd_model.sample_denoiser(sample_batch)
        vbd_output = predictions["denoised_trajs"].to("cpu").detach().cpu().numpy()
        
        # Get agent IDs
        agent_indices = sample_batch["agents_id"][0]  # Assuming batch size 1
        
        # Store predictions in dictionary
        valid_mask = agent_indices >= 0
        valid_agent_indices = agent_indices[valid_mask]
        
        for i, agent_idx in enumerate(valid_agent_indices):
            agent_id = env.sim.absolute_self_observation_tensor().to_torch()[0, agent_idx, -1].item()
            
            # Store predictions for this window
            for t in range(window_size):
                if start_idx + t < total_steps:
                    # Store position (x, y), heading (yaw), velocity (vx, vy)
                    vbd_trajectories[agent_id][start_idx + t] = [
                        float(vbd_output[0, i, t, 0]),  # x
                        float(vbd_output[0, i, t, 1]),  # y
                        float(vbd_output[0, i, t, 2]),  # yaw
                        float(vbd_output[0, i, t, 3]),  # vx
                        float(vbd_output[0, i, t, 4])   # vy
                    ]
        
        # Advance the environment by window_size steps
        if start_idx + window_size < total_steps:
            for _ in range(window_size):
                # Get expert actions for the current step
                expert_actions, _, _, _ = env.get_expert_actions()
                env.step_dynamics(expert_actions[:, :, 0, :])
    
    # Add VBD trajectories to the JSON file
    for obj in data["objects"]:
        agent_id = obj["id"]
        if agent_id in vbd_trajectories:
            # Add a new key for VBD trajectories
            obj["vbd_trajectories"] = vbd_trajectories[agent_id]
    
    # Write the updated JSON file
    output_path = os.path.join(output_dir, os.path.basename(json_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="VBD Amortization Script (CPU-only version)")
    parser.add_argument("--model_path", required=True, help="Path to the VBD model checkpoint")
    parser.add_argument("--input_dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory for output JSON files")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--window_size", type=int, default=10, help="Size of sliding window")
    parser.add_argument("--total_steps", type=int, default=91, help="Total number of steps to generate")
    parser.add_argument("--single_file", help="Process only a single file (optional)")
    args = parser.parse_args()
    
    # Always use CPU device
    device = "cpu"
    print(f"Loading VBD model on {device}...")
    vbd_model = load_vbd_model(args.model_path, device)
    
    if args.single_file:
        # Process just one file
        json_files = [Path(args.single_file)]
    else:
        # Find all JSON files in the input directory
        json_files = list(Path(args.input_dir).glob("**/*.json"))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for json_path in tqdm(json_files):
        try:
            output_path = process_file(
                str(json_path),
                vbd_model,
                args.output_dir,
                args.batch_size,
                args.window_size,
                args.total_steps
            )
            print(f"Processed {json_path} -> {output_path}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("VBD amortization complete!")

if __name__ == "__main__":
    main()