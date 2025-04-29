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
import trimesh

logging.basicConfig(level=logging.INFO)

def _filter_small_segments(segments, min_length=1e-6):
    """Filter out segments that are too short."""
    valid_segments = []
    for segment in segments:
        start, end = segment
        length = np.linalg.norm(np.array(end) - np.array(start))
        if length >= min_length:
            valid_segments.append(segment)
    return valid_segments


def _generate_mesh(segments, height=2.0, width=0.2):
    """Generate a mesh from line segments."""
    segments = np.array(segments, dtype=np.float64)
    starts, ends = segments[:, 0, :], segments[:, 1, :]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / lengths

    # Create the base box mesh with the height along the z-axis
    base_box = trimesh.creation.box(extents=[1.0, width, height])
    base_box.apply_translation([0.5, 0, 0])  # Align box's origin to its start
    z_axis = np.array([0, 0, 1])
    angles = np.arctan2(
        unit_directions[:, 1], unit_directions[:, 0]
    )  # Rotation in the XY plane

    rectangles = []
    lengths = lengths.flatten()

    for i, (start, length, angle) in enumerate(zip(starts, lengths, angles)):
        # Copy the base box and scale to match segment length
        scaled_box = base_box.copy()
        scaled_box.apply_scale([length, 1.0, 1.0])

        # Apply rotation around the z-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle, z_axis
        )
        scaled_box.apply_transform(rotation_matrix)

        # Translate the box to the segment's starting point
        scaled_box.apply_translation(start)

        rectangles.append(scaled_box)

    # Concatenate all boxes into a single mesh
    mesh = trimesh.util.concatenate(rectangles)
    return mesh


def _create_agent_box_mesh(position, heading, length, width, height):
    """Create a box mesh for an agent at a given position and orientation.
    
    Args:
        position (list): [x, y, z] position
        heading (float): yaw angle in radians
        length (float): length of the box
        width (float): width of the box 
        height (float): height of the box
        
    Returns:
        trimesh.Trimesh: Box mesh positioned and oriented correctly
    """
    # Create box centered at origin
    box = trimesh.creation.box(extents=[length, width, height])
    
    # Rotate box to align with heading
    z_axis = np.array([0, 0, 1])
    rotation_matrix = trimesh.transformations.rotation_matrix(heading, z_axis)
    box.apply_transform(rotation_matrix)
    
    # Move box to position
    box.apply_translation(position)
    
    return box


def calculate_trajectory_length(positions, valid_mask, start_idx, end_idx=90):
    """
    Calculate the sum of valid trajectory segments from start_idx to end_idx.
    
    Args:
        positions: List of position dictionaries with x, y, z coordinates
        valid_mask: List of boolean values indicating valid timesteps
        start_idx: Starting index for calculation
        end_idx: Ending index for calculation
        
    Returns:
        float: Sum of valid trajectory segments length
    """
    # Limit end_idx to the maximum available index
    end_idx = min(end_idx, len(positions) - 1)
    
    # Initialize total length
    total_length = 0.0
    
    # Iterate through positions from start_idx to end_idx
    for i in range(start_idx, end_idx):
        # Check if both current and next positions are valid
        if valid_mask[i] and valid_mask[i + 1]:
            # Calculate distance between consecutive valid positions
            current_pos = np.array([positions[i]['x'], positions[i]['y'], positions[i]['z']])
            next_pos = np.array([positions[i + 1]['x'], positions[i + 1]['y'], positions[i + 1]['z']])
                
            segment_length = np.linalg.norm(next_pos - current_pos)
            total_length += segment_length
            
    return total_length


def process_scene(args):
    """Process a single scene file to detect parked vehicles."""
    filepath, init_steps, threshold = args
    try:
        with open(filepath, 'r') as f:
            scene = json.load(f)

        # Initialize counters for this scene
        valid_vehicles = 0
        colliding_vehicles = 0
        parked_vehicles = 0
        moving_vehicles = 0

        # Extract road data for collision checking
        roads = scene['roads']
        edge_segments = []

        # Collect road edge segments for collision checking
        for road in roads:
            if road["type"] == "road_edge":
                edge_vertices = [[r["x"], r["y"], r["z"]] for r in road["geometry"]]
                edge_segments.extend([
                    [edge_vertices[i], edge_vertices[i + 1]]
                    for i in range(len(edge_vertices) - 1)
                ])

        # Generate road edge mesh
        edge_segments = _filter_small_segments(edge_segments)
        edge_mesh = _generate_mesh(edge_segments)

        # Create collision managers
        road_collision_manager = trimesh.collision.CollisionManager()
        road_collision_manager.add_object("road_edges", edge_mesh)
        vehicle_collision_manager = trimesh.collision.CollisionManager()

        # First, collect all valid vehicles at init_steps
        valid_vehicle_objects = []
        for obj in scene['objects']:
            # Check if object is a vehicle and valid at init_steps
            if (obj['type'] == 'vehicle' and 
                init_steps < len(obj['valid']) and 
                obj['valid'][init_steps]):
                
                valid_vehicles += 1
                valid_vehicle_objects.append(obj)
                
                # Create vehicle box at init_steps
                position = [
                    obj['position'][init_steps]['x'],
                    obj['position'][init_steps]['y'],
                    obj['position'][init_steps]['z']
                ]
                heading = obj['heading'][init_steps]
                vehicle_box = _create_agent_box_mesh(
                    position,
                    heading,
                    obj['length'],
                    obj['width'],
                    obj['height']
                )
                vehicle_collision_manager.add_object(str(obj['id']), vehicle_box)

        # Check for collisions with other vehicles
        _, vehicle_collision_pairs = vehicle_collision_manager.in_collision_internal(return_names=True)
        
        # Check for collisions with road edges
        _, road_collision_pairs = vehicle_collision_manager.in_collision_other(
            road_collision_manager, return_names=True
        )
        
        # Combine all colliding vehicle IDs
        colliding_vehicle_ids = set()
        
        # Add vehicles that collide with each other
        for v1, v2 in vehicle_collision_pairs:
            colliding_vehicle_ids.add(v1)
            colliding_vehicle_ids.add(v2)
        
        # Add vehicles that collide with road edges
        for v_id, _ in road_collision_pairs:
            colliding_vehicle_ids.add(v_id)
            
        colliding_vehicles = len(colliding_vehicle_ids)
        
        # Process each colliding vehicle to determine if parked or moving
        for obj in valid_vehicle_objects:
            if str(obj['id']) in colliding_vehicle_ids:
                # Calculate trajectory length from init_steps to 90 or last timestep
                trajectory_length = calculate_trajectory_length(
                    obj['position'], 
                    obj['valid'], 
                    init_steps,
                    min(90, len(obj['valid']) - 1)
                )
                
                # Classify as parked or moving based on threshold
                if trajectory_length < threshold:
                    parked_vehicles += 1
                else:
                    moving_vehicles += 1

        return filepath, (valid_vehicles, colliding_vehicles, parked_vehicles, moving_vehicles)

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return filepath, None


def process_directory(args):
    """Process all JSON files in directory."""
    input_dir = Path(args.input_dir)
    num_workers = args.num_workers
    init_steps = args.init_steps
    threshold = args.threshold

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
    total_valid_vehicles = np.int64(0)
    total_colliding_vehicles = np.int64(0)
    total_parked_vehicles = np.int64(0)
    total_moving_vehicles = np.int64(0)
    
    # Process files in batches
    for i in range(0, len(json_files), int(batch_size)):
        batch = json_files[i:i + int(batch_size)]
        
        # Process batch in parallel
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_scene, [(str(f), init_steps, threshold) for f in batch]),
                total=len(batch),
                desc=f"Processing batch {i//int(batch_size) + 1}"
            ))
        
        # Count results
        for filepath, counts in results:
            if counts is not None:
                valid_vehicles, colliding_vehicles, parked_vehicles, moving_vehicles = counts
                total_processed += 1
                total_valid_vehicles += valid_vehicles
                total_colliding_vehicles += colliding_vehicles
                total_parked_vehicles += parked_vehicles
                total_moving_vehicles += moving_vehicles
    
    # Calculate percentages using float64 for precision
    parked_percentage = (float(total_parked_vehicles) / float(total_colliding_vehicles) * 100) if total_colliding_vehicles > 0 else 0.0
    moving_percentage = (float(total_moving_vehicles) / float(total_colliding_vehicles) * 100) if total_colliding_vehicles > 0 else 0.0
    
    logging.info(f"Processing complete!")
    logging.info(f"Total files processed: {total_processed:,d}")
    logging.info(f"Total vehicles valid at t={init_steps}: {total_valid_vehicles:,d}")
    logging.info(f"Total vehicles in collision at t={init_steps}: {total_colliding_vehicles:,d}")
    logging.info(f"Total parked vehicles: {total_parked_vehicles:,d} ({parked_percentage:.2f}%)")
    logging.info(f"Total moving vehicles: {total_moving_vehicles:,d} ({moving_percentage:.2f}%)")
    
    # Save results to a JSON file for future reference
    results = {
        "total_files_processed": int(total_processed),
        "total_valid_vehicles": int(total_valid_vehicles),
        "total_colliding_vehicles": int(total_colliding_vehicles),
        "total_parked_vehicles": int(total_parked_vehicles),
        "total_moving_vehicles": int(total_moving_vehicles),
        "parked_percentage": float(parked_percentage),
        "moving_percentage": float(moving_percentage)
    }
    
    with open('parked_vehicle_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze vehicle parking behavior in JSON files"
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
        "--init_steps",
        type=int,
        default=10,
        help="Initial timestep to check for vehicle validity and collisions (default: 10)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Threshold for trajectory length to classify a vehicle as moving (default: 0.2)"
    )

    args = parser.parse_args()
    process_directory(args)