import os
import json
import logging
import psutil
import argparse
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def check_3d_structures(edge_points, tolerance=0.1):
    """Check for 3D structures using point-wise comparisons."""
    if len(edge_points) == 0:
        return False

    edge_points = np.array(edge_points)
    xy_points = edge_points[:, :2]
    
    # Process in chunks for memory efficiency
    chunk_size = 1000
    for i in range(0, len(xy_points), chunk_size):
        chunk = xy_points[i:i + chunk_size]
        # Calculate distances between current chunk and all points
        dists = np.linalg.norm(chunk[:, np.newaxis] - xy_points, axis=2)
        potential_pairs = np.where((dists < tolerance) & (dists > 0))
        
        # Check z-values for identified pairs
        for p1, p2 in zip(*potential_pairs):
            p1_idx = i + p1  # Adjust index for chunking
            if abs(edge_points[p1_idx, 2] - edge_points[p2, 2]) > tolerance:
                return True
    
    return False

def process_scene(args):
    """Process a single scene file."""
    filepath, tolerance = args
    try:
        with open(filepath, 'r') as f:
            scene = json.load(f)

        # Collect edge points
        edge_points = []
        for road in scene['roads']:
            if road["type"] == "road_edge":
                edge_points.extend([[r["x"], r["y"], r["z"]] for r in road["geometry"]])

        # Check for 3D structures
        if check_3d_structures(edge_points, tolerance):
            os.remove(filepath)
            return filepath, True
        return filepath, False

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return filepath, None

def process_directory(args):
    """Process all JSON files in directory."""
    input_dir = Path(args.input_dir)
    tolerance = args.tolerance
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
    # Assuming each scene takes about 1MB
    batch_size = min(1000 * usable_memory, len(json_files))
    
    # Process files in batches
    total_processed = 0
    total_3d = 0
    
    for i in range(0, len(json_files), int(batch_size)):
        batch = json_files[i:i + int(batch_size)]
        
        # Process batch in parallel
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_scene, [(str(f), tolerance) for f in batch]),
                total=len(batch),
                desc=f"Processing batch {i//int(batch_size) + 1}"
            ))
        
        # Count results
        for filepath, is_3d in results:
            if is_3d is not None:
                total_processed += 1
                if is_3d:
                    total_3d += 1
                    logging.info(f"Removed 3D scene: {filepath}")
    
    logging.info(f"Processing complete!")
    logging.info(f"Total files processed: {total_processed}")
    logging.info(f"Total 3D scenes removed: {total_3d}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSON files and remove those containing 3D structures"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing JSON files to process"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Tolerance for 3D structure detection (default: 0.2)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: number of CPU cores)"
    )

    args = parser.parse_args()
    process_directory(args)