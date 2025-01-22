import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict
import itertools
import shutil
import os

def extract_road_edges(json_data: Dict) -> np.ndarray:
    """
    Extract road edge points from JSON data and return as numpy array.
    Returns array of shape (N, 3) where N is number of points.
    """
    points = []
    for road in json_data["roads"]:
        if road["type"] == "road_edge":
            geometry = road["geometry"]
            points.extend((p["x"], p["y"], p["z"]) for p in geometry)
    return np.array(points)

def find_overlapping_points(points: np.ndarray, tolerance: float = 0.1) -> List[Tuple]:
    """
    Find points that overlap in x,y coordinates but have different z values.
    Uses vectorized operations for efficiency.
    
    Args:
        points: Numpy array of shape (N, 3) containing x,y,z coordinates
        tolerance: Maximum distance in x,y plane to consider points as overlapping
    
    Returns:
        List of tuples containing indices of overlapping points
    """
    # Remove error values
    valid_mask = points[:, 2] != -10000
    points = points[valid_mask]
    
    if len(points) == 0:
        return []
    
    # Calculate pairwise distances in xy plane
    xy_points = points[:, :2]
    dists = np.linalg.norm(xy_points[:, np.newaxis] - xy_points, axis=2)
    
    # Find pairs within tolerance in xy plane
    potential_pairs = np.where((dists < tolerance) & (dists > 0))
    
    # Check z-values for identified pairs
    overlapping = []
    for i, j in zip(*potential_pairs):
        if abs(points[i, 2] - points[j, 2]) > tolerance:
            overlapping.append((i, j))
    
    return overlapping

def process_single_file(args: Tuple[Path, Path, Path]) -> Tuple[str, bool]:
    """
    Process a single JSON file and move it to appropriate directory based on content.
    Returns tuple of (file_path, has_overlap, is_driveway, was_moved_3d, was_moved_driveway)
    """
    file_path, target_3d_dir = args
    
    with open(file_path) as f:
        data = json.load(f)
    
    was_moved_3d = False
    points = extract_road_edges(data)
    overlapping = find_overlapping_points(points)
    if len(overlapping) > 0:
        # Move the file
        shutil.move(str(file_path), str(target_3d_dir / file_path.name))
        was_moved_3d = True

    return str(file_path), was_moved_3d

def main(source_dir: str, target_3d_dir: str, num_processes: int = cpu_count()):
    """
    Process all JSON files in directory in parallel and move files to appropriate directories.
    Args:
        source_dir: Path to directory containing JSON files
        target_3d_dir: Path to directory where 3D files should be moved
        target_driveway_dir: Path to directory where driveway files should be moved
        num_processes: Number of parallel processes to use
    """
    source_path = Path(source_dir)
    target_3d_path = Path(target_3d_dir)

    # Create target directories at the start
    target_3d_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(source_path.glob("*.json"))
    # Create list of tuples containing (file_path, target_3d_dir, target_driveway_dir)
    process_args = [(f, target_3d_path) for f in json_files]
    
    with Pool(num_processes) as pool:
        results = pool.map(process_single_file, process_args)
    
    # Count moved files and print results
    moved_3d_count = 0
    
    for file_path, was_moved_3d in results:
        if was_moved_3d:
            print(f"Moved {file_path} to 3D directory")
            moved_3d_count += 1
    
    print(f"\nTotal files moved to 3D directory: {moved_3d_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_directory_path> <target_3d_directory_path>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])