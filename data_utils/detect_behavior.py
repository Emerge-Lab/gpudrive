import os
import json
import logging
import psutil
import argparse
import math
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def check_uturn(headings, valid_mask):
    """
    Check if a vehicle makes a U-turn by comparing heading angles.
    Args:
        headings: List of heading angles in radians
        valid_mask: List of boolean values indicating valid timesteps
    Returns:
        bool: True if U-turn detected
    """
    # Convert 150 degrees to radians (150 * pi/180)
    angle_threshold = 2.618  # approximately 150 degrees in radians
    
    # Get first valid heading
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if not valid_indices:
        return False
        
    first_valid_idx = valid_indices[0]
    first_heading = headings[first_valid_idx]
    
    # Check subsequent valid headings
    for i in valid_indices[1:]:
        angle_diff = abs(headings[i] - first_heading)
        # Normalize angle difference to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff) > angle_threshold:
            return True
            
    return False

def check_reversing(headings, velocities, valid_mask, min_timesteps=10):
    """
    Check if a vehicle reverses by comparing velocity direction with heading.
    Args:
        headings: List of heading angles in radians
        velocities: List of dictionaries containing 'x' and 'y' velocities
        valid_mask: List of boolean values indicating valid timesteps
        min_timesteps: Minimum number of consecutive timesteps required for reversing
    Returns:
        bool: True if sustained reversing detected
    """
    # Convert angle range to radians (150 to 210 degrees)
    min_angle = 2.618  # 150 degrees
    
    consecutive_reverse = 0
    
    for i, valid in enumerate(valid_mask):
        if not valid:
            consecutive_reverse = 0
            continue
            
        # Calculate velocity direction
        vx = velocities[i]['x']
        vy = velocities[i]['y']
        
        # Skip stationary moments
        if abs(vx) < 0.1 and abs(vy) < 0.1:
            consecutive_reverse = 0
            continue
            
        velocity_angle = math.atan2(vy, vx)
        heading = headings[i]
        
        # Calculate angle between velocity and heading
        angle_diff = velocity_angle - heading
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Check if velocity is in the reverse cone
        if min_angle <= abs(angle_diff):
            consecutive_reverse += 1
            if consecutive_reverse >= min_timesteps:
                return True
        else:
            consecutive_reverse = 0
            
    return False

def process_scene(args):
    """Process a single scene file."""
    filepath, min_reverse_timesteps = args
    try:
        with open(filepath, 'r') as f:
            scene = json.load(f)

        uturn_count = np.int64(0)
        reverse_count = np.int64(0)
        total_agents = np.int64(0)

        # Process each object
        for obj in scene['objects']:
            # Check if object is a vehicle or cyclist and not an expert
            if (obj['type'] in ['vehicle', 'cyclist'] and 
                not obj.get('mark_as_expert', False)):
                
                total_agents += 1
                
                # Get valid mask and corresponding headings/velocities
                valid_mask = obj['valid']
                headings = obj['heading']
                velocities = obj['velocity']
                
                # Check for U-turn
                if check_uturn(headings, valid_mask):
                    uturn_count += 1
                    
                # Check for reversing
                if check_reversing(headings, velocities, valid_mask, min_reverse_timesteps):
                    reverse_count += 1

        return filepath, (total_agents, uturn_count, reverse_count)

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return filepath, None

def process_directory(args):
    """Process all JSON files in directory."""
    input_dir = Path(args.input_dir)
    num_workers = args.num_workers

    # Get all JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        logging.error(f"No JSON files found in {input_dir}")
        return

    logging.info(f"Found {len(json_files)} JSON files to process")

    # Calculate batch size based on available memory
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available / (1024**3)  # Convert to GB
    usable_memory = int(available_memory * 0.9)  # Use 90% of available memory
    batch_size = min(1000 * usable_memory, len(json_files))
    
    # Initialize counters using numpy int64 to handle large numbers
    total_processed = np.int64(0)
    total_agents = np.int64(0)
    total_uturns = np.int64(0)
    total_reverses = np.int64(0)
    
    # Process files in batches
    for i in range(0, len(json_files), int(batch_size)):
        batch = json_files[i:i + int(batch_size)]
        
        # Process batch in parallel
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_scene, [(str(f), args.min_reverse_timesteps) for f in batch]),
                total=len(batch),
                desc=f"Processing batch {i//int(batch_size) + 1}"
            ))
        
        # Count results
        for filepath, counts in results:
            if counts is not None:
                agents, uturns, reverses = counts
                total_processed += 1
                total_agents += agents
                total_uturns += uturns
                total_reverses += reverses
    
    # Calculate percentages using float64 for precision
    uturn_percentage = (float(total_uturns) / float(total_agents) * 100) if total_agents > 0 else 0.0
    reverse_percentage = (float(total_reverses) / float(total_agents) * 100) if total_agents > 0 else 0.0
    
    logging.info(f"Processing complete!")
    logging.info(f"Total files processed: {total_processed:,d}")
    logging.info(f"Total non-expert agents: {total_agents:,d}")
    logging.info(f"Total U-turns: {total_uturns:,d} ({uturn_percentage:.2f}%)")
    logging.info(f"Total reversing: {total_reverses:,d} ({reverse_percentage:.2f}%)")
    
    # Also save results to a JSON file for future reference
    results = {
        "total_files_processed": int(total_processed),
        "total_non_expert_agents": int(total_agents),
        "total_uturns": int(total_uturns),
        "total_reversing": int(total_reverses),
        "uturn_percentage": float(uturn_percentage),
        "reverse_percentage": float(reverse_percentage)
    }
    
    with open('vehicle_behavior_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze vehicle behaviors in JSON files"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing JSON files to process"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: number of CPU cores)"
    )
    parser.add_argument(
        "--min_reverse_timesteps",
        type=int,
        default=3,
        help="Minimum number of consecutive timesteps required for reversing (default: 3)"
    )

    args = parser.parse_args()
    process_directory(args)