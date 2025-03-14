import argparse
import json
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import tqdm

def is_valid_json_structure(file_path):
    """Check if the JSON file has the expected structure."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            
            # Check if required keys exist in the top-level structure
            if not all(key in data for key in ["name", "objects", "roads", "tl_states"]):
                return False
                
            # Check if "objects" is a list and contains dictionaries with the correct structure
            if not isinstance(data["objects"], list) or not all(
                isinstance(obj, dict) and "position" in obj and "type" in obj
                for obj in data["objects"]
            ):
                return False
                
            # Check if "roads" is a list of dictionaries with required "geometry"
            if not isinstance(data["roads"], list) or not all(
                isinstance(road, dict) and "geometry" in road
                for road in data["roads"]
            ):
                return False
                
            # Check that each "geometry" in "roads" has valid "x" and "y" coordinates
            for road in data["roads"]:
                if not all(
                    isinstance(geo, dict) and "x" in geo and "y" in geo
                    for geo in road.get("geometry", [])
                ):
                    return False
                    
            return True
    except (json.JSONDecodeError, ValueError, IOError):
        return False

def process_file(args):
    """
    Validate JSON file and handle it according to the operation mode.
    
    Args:
        args (tuple): (source_path, target_dir, should_move)
            - source_path: Path to the source file
            - target_dir: Path to target directory (if moving)
            - should_move: Boolean indicating if file should be moved if valid
    Returns:
        tuple: (str, bool) - (file path, whether file was valid)
    """
    source_path, target_dir, should_move = args
    
    # First validate the JSON
    if not is_valid_json_structure(source_path):
        try:
            source_path.unlink()  # Delete invalid file
            return str(source_path), False
        except Exception as e:
            print(f"Error deleting invalid file {source_path}: {e}")
            return str(source_path), False
    
    # If valid and should_move is True, move the file
    if should_move and target_dir:
        try:
            target_path = Path(target_dir) / source_path.name
            shutil.move(str(source_path), str(target_path))
        except Exception as e:
            print(f"Error moving file {source_path}: {e}")
            return str(source_path), False
    
    return str(source_path), True

def process_directory(dataset_dir, num_workers=None):
    """
    Process all JSON files in a directory, automatically handling group extraction if needed.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        num_workers (int, optional): Number of processes to use. Defaults to CPU count.
    Returns:
        tuple: (int, int) - (valid_files, invalid_files)
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.is_dir():
        print(f"Directory {dataset_dir} does not exist, skipping...")
        return 0, 0
    
    # Check for group directories
    group_dirs = [d for d in dataset_path.iterdir() 
                 if d.is_dir() and d.name.startswith("group_")]
    
    # Collect all files that need to be processed
    all_files = []
    
    if group_dirs:
        # Found group directories - will extract files from them
        print(f"\nFound {len(group_dirs)} group directories in {dataset_dir}")
        for group_dir in sorted(group_dirs):
            files = list(group_dir.glob("*.json"))
            all_files.extend([(file, dataset_path, True) for file in files])
    
    # Always check for JSON files in the main directory as well
    main_dir_files = [f for f in dataset_path.glob("*.json") 
                     if not any(g.name in str(f) for g in group_dirs)]
    all_files.extend([(file, None, False) for file in main_dir_files])
    
    total_files = len(all_files)
    if total_files == 0:
        print(f"No JSON files found in {dataset_dir}")
        return 0, 0
    
    print(f"Total files to process: {total_files}")
    
    # Use all available CPUs if num_workers is not specified
    if num_workers is None:
        num_workers = cpu_count()
    
    # Track statistics
    valid_files = 0
    invalid_files = 0
    
    # Create a pool of workers and process files in parallel
    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar
        results = list(tqdm.tqdm(
            pool.imap_unordered(process_file, all_files),
            total=total_files,
            desc=f"Processing files from {dataset_dir}"
        ))
        
        # Count valid and invalid files
        for _, is_valid in results:
            if is_valid:
                valid_files += 1
            else:
                invalid_files += 1
    
    # If we found group directories, try to remove empty ones after processing
    if group_dirs:
        for group_dir in group_dirs:
            try:
                group_dir.rmdir()
            except OSError:
                print(f"Warning: Could not remove directory {group_dir} - it may not be empty")
    
    print(f"\nCompleted processing {dataset_dir}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files deleted: {invalid_files}")
    
    return valid_files, invalid_files

def process_all_directories(num_workers=None):
    """Process all dataset directories (training, testing, validation)."""
    directories = [
        "data/processed/training",
        "data/processed/testing",
        "data/processed/validation"
    ]
    
    total_valid = 0
    total_invalid = 0
    
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        valid, invalid = process_directory(directory, num_workers)
        total_valid += valid
        total_invalid += invalid
    
    print("\nOverall Statistics:")
    print(f"Total valid files across all directories: {total_valid}")
    print(f"Total invalid files deleted: {total_invalid}")
    print(f"Total files processed: {total_valid + total_invalid}")

def main():
    parser = argparse.ArgumentParser(
        description="Process JSON files in dataset directories, validating their structure and "
                  "automatically extracting from group directories if they exist. "
                  "Invalid files are deleted. "
                  'Use "all" to process training, testing, and validation directories.'
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="all",
        help='Path to the dataset directory or "all" for processing all directories'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of processes to use (defaults to number of CPU cores)",
        default=cpu_count()
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset_dir.lower() == "all":
            process_all_directories(args.num_workers)
        else:
            process_directory(args.dataset_dir, args.num_workers)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
