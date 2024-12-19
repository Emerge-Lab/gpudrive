import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import tqdm

def move_file(args):
    """
    Move a single file to its target location.
    
    Args:
        args (tuple): (source_path, target_dir)
    """
    source_path, target_dir = args
    target_path = Path(target_dir) / source_path.name
    shutil.move(str(source_path), str(target_path))
    return str(source_path)

def extract_groups(dataset_dir, num_workers=None):
    """
    Extract all files from group directories back to the parent directory in parallel.
    
    Args:
        dataset_dir (str): Path to the dataset directory containing group folders
        num_workers (int, optional): Number of processes to use. Defaults to CPU count.
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.is_dir():
        raise ValueError(f"Directory {dataset_dir} does not exist")
    
    # Find all group directories
    group_dirs = [d for d in dataset_path.iterdir() 
                 if d.is_dir() and d.name.startswith("group_")]
    
    if not group_dirs:
        print(f"No group directories found in {dataset_dir}!")
        return
    
    print(f"\nProcessing {dataset_dir}")
    print(f"Found {len(group_dirs)} group directories")
    
    # Collect all files that need to be moved
    all_files = []
    for group_dir in sorted(group_dirs):
        files = list(group_dir.glob("*.json"))
        all_files.extend([(file, dataset_path) for file in files])
    
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")
    
    # Use all available CPUs if num_workers is not specified
    if num_workers is None:
        num_workers = cpu_count()
    
    # Create a pool of workers and process files in parallel
    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar
        list(tqdm.tqdm(
            pool.imap_unordered(move_file, all_files),
            total=total_files,
            desc=f"Moving files from {dataset_dir}"
        ))
    
    # Remove empty group directories
    for group_dir in group_dirs:
        group_dir.rmdir()
    
    print(f"Completed {dataset_dir}")
    print(f"Total files processed: {total_files}")

def process_default_directory(num_workers=None):
    """
    Process the default training, testing, and validation directories in parallel.
    
    Args:
        num_workers (int, optional): Number of processes to use per directory.
    """
    default_dir = "data/processed/training"
    # Process each directory with its own pool of workers
    try:
        extract_groups(default_dir, num_workers)
    except Exception as e:
        print(f"Error processing {default_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract files from group directory back to parent directory in parallel. "
                  "If no directory is specified, processes data/processed/training by default."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",  # Makes the argument optional
        help="Path to the dataset directory containing group folders"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of processes to use (defaults to number of CPU cores)",
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset_dir:
            # Process single specified directory
            extract_groups(args.dataset_dir, args.num_workers)
        else:
            # Process default directories
            process_default_directory(args.num_workers)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())