import os
# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
import glob
import argparse
import random
import json
import pickle  # Added for loading pickle files
import multiprocessing
from tqdm import tqdm
from functools import partial
from waymo_open_dataset.protos import scenario_pb2

def extract_scene_from_tfrecord(tfrecord_path, scene_id, output_path):
    """
    Extract a specific scene from a TFRecord file based on its ID.
    
    Args:
        tfrecord_path (str): Path to the TFRecord file
        scene_id (str): ID of the scene to extract
        output_path (str): Path to save the extracted scene
    
    Returns:
        bool: True if scene was found and extracted, False otherwise
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    found = False
    
    for tf_data in dataset:
        tf_data_bytes = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data_bytes))
        
        # Check if this is the scene we're looking for
        if scenario.scenario_id == scene_id:
            with tf.io.TFRecordWriter(output_path) as file_writer:
                file_writer.write(tf_data_bytes)
            found = True
            break
    
    return found

def process_single_scene(json_file, dataset, data_dir, tfrecord_dir, json_save_dir, tfrecord_save_dir):
    """
    Process a single scene file.
    
    Args:
        json_file (str): Path to the JSON file
        dataset (str): Name of the dataset
        data_dir (str): Base directory containing JSON files
        tfrecord_dir (str): Directory containing TFRecord files
        json_save_dir (str): Directory to save processed JSON files
        tfrecord_save_dir (str): Directory to save processed TFRecord files
        
    Returns:
        tuple: (json_file, success, message)
    """
    try:
        # Extract prefix and suffix from filename
        filename = os.path.basename(json_file)
        name_parts = os.path.splitext(filename)[0].split('_', 1)  # Split on the first underscore only
        if len(name_parts) != 2:
            return json_file, False, f"Invalid filename format: {filename}, expected <prefix>_<suffix>.json"
        
        prefix, suffix = name_parts
        
        # Find corresponding TFRecord file
        tfrecord_path = os.path.join(tfrecord_dir, dataset, f"{dataset}.{prefix}")
        
        if not os.path.exists(tfrecord_path):
            return json_file, False, f"TFRecord file not found: {tfrecord_path}"
        
        # Copy the JSON file to the output directory with new name
        json_output_path = os.path.join(json_save_dir, f"{suffix}.json")
        
        # Copy the JSON file content
        with open(json_file, 'r') as f_in:
            json_data = json.load(f_in)
        
        with open(json_output_path, 'w') as f_out:
            json.dump(json_data, f_out)
        
        # Extract the scene from the TFRecord file
        tfrecord_output_path = os.path.join(tfrecord_save_dir, f"{suffix}.tfrecords")
        found = extract_scene_from_tfrecord(tfrecord_path, suffix, tfrecord_output_path)
        
        if not found:
            # Remove the JSON file if the TFRecord extraction failed
            os.remove(json_output_path)
            return json_file, False, f"Scene {suffix} not found in TFRecord file: {tfrecord_path}"
        
        return json_file, True, "Successfully processed"
    
    except Exception as e:
        return json_file, False, f"Error processing {json_file}: {str(e)}"

def find_json_by_suffix(data_path, suffix):
    """
    Find a JSON file with a specific suffix in a directory.
    
    Args:
        data_path (str): Directory path to search in
        suffix (str): Suffix to look for
        
    Returns:
        str or None: Path to the JSON file if found, None otherwise
    """
    json_files = glob.glob(os.path.join(data_path, '*.json'))
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        name_parts = os.path.splitext(filename)[0].split('_', 1)
        if len(name_parts) == 2 and name_parts[1] == suffix:
            return json_file
    
    return None

def process(dataset, num_scenes, data_dir, tfrecord_dir, save_dir, num_workers, scene_ids_pkl=None):
    """
    Process a specific dataset, selecting random scenes and extracting their TFRecords.
    
    Args:
        dataset (str): Name of the dataset to process
        num_scenes (int): Number of scenes to randomly select
        data_dir (str): Directory containing JSON files
        tfrecord_dir (str): Directory containing TFRecord files
        save_dir (str): Directory to save processed files
        num_workers (int): Number of worker processes to use
        scene_ids_pkl (str): Path to pickle file containing scene IDs
    """
    # Set up paths
    data_path = os.path.join(data_dir, dataset)
    
    # Create output directories
    json_save_dir = os.path.join(save_dir, f"{dataset}", "json")
    tfrecord_save_dir = os.path.join(save_dir, f"{dataset}", "tfrecord")
    os.makedirs(json_save_dir, exist_ok=True)
    os.makedirs(tfrecord_save_dir, exist_ok=True)
    
    selected_files = []
    
    # If pickle file with scene IDs is provided, use those IDs instead of random selection
    if scene_ids_pkl:
        try:
            with open(scene_ids_pkl, 'rb') as f:
                scene_ids = pickle.load(f)
            
            print(f"Loaded {len(scene_ids)} scene IDs from {scene_ids_pkl}")
            
            # Find corresponding JSON files for each scene ID
            found_files = []
            not_found_ids = []
            
            for suffix in scene_ids:
                json_file = find_json_by_suffix(data_path, suffix)
                if json_file:
                    found_files.append(json_file)
                else:
                    not_found_ids.append(suffix)
            
            selected_files = found_files
            
            print(f"Found {len(selected_files)} matching JSON files, {len(not_found_ids)} scene IDs not found")
            if not_found_ids and len(not_found_ids) < 10:
                print(f"IDs not found: {', '.join(not_found_ids)}")
            elif not_found_ids:
                print(f"First 5 IDs not found: {', '.join(not_found_ids[:5])}, and {len(not_found_ids) - 5} more")
                
        except Exception as e:
            print(f"Error loading scene IDs from {scene_ids_pkl}: {str(e)}")
            return
    else:
        # Use random selection as before
        json_files = glob.glob(os.path.join(data_path, '*.json'))
        n_files = len(json_files)
        
        if n_files == 0:
            print(f"No JSON files found in {data_path}")
            return
        
        print(f"Found {n_files} files in {dataset} dataset")
        num_to_select = min(num_scenes, n_files)
        selected_files = random.sample(json_files, num_to_select)
    
    if not selected_files:
        print(f"No files to process for {dataset}")
        return
    
    print(f"Processing {len(selected_files)} files from {dataset} using {num_workers} workers")
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_single_scene,
        dataset=dataset,
        data_dir=data_dir,
        tfrecord_dir=tfrecord_dir,
        json_save_dir=json_save_dir,
        tfrecord_save_dir=tfrecord_save_dir
    )
    
    # Create a pool of workers and process the files
    # Adjust number of workers to not exceed CPU count
    num_workers = min(num_workers, multiprocessing.cpu_count(), len(selected_files))
    
    # If only one worker, process sequentially (avoiding multiprocessing overhead)
    if num_workers == 1:
        results = []
        for json_file in tqdm(selected_files, desc=f"Processing {dataset}"):
            results.append(process_func(json_file))
    else:
        # Use multiprocessing for parallel processing
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, selected_files),
                total=len(selected_files),
                desc=f"Processing {dataset}"
            ))
    
    # Count successes and failures
    successes = sum(1 for _, success, _ in results if success)
    failures = len(results) - successes
    
    print(f"Finished processing {dataset}: {successes} succeeded, {failures} failed")
    
    # Print some failures if there are any (limit to 5 to avoid flooding console)
    if failures > 0:
        print("Some failures occurred:")
        failure_count = 0
        for _, success, message in results:
            if not success:
                print(f"  - {message}")
                failure_count += 1
                if failure_count >= 5 and failures > 5:
                    print(f"  ... and {failures - 5} more failures")
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed/')
    parser.add_argument('--save_dir', type=str, default='data/processed/wosac/')
    parser.add_argument('--tfrecord_dir', type=str, default='data/raw/')
    parser.add_argument('--dataset', type=str, default='validation')
    parser.add_argument('--num_scenes', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--scene_ids_pkl', type=str, default=None, help='Path to pickle file containing scene IDs to process')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f'Processing data from {args.data_dir} and Saving to {args.save_dir}')
    
    if args.scene_ids_pkl:
        print(f'Using scene IDs from {args.scene_ids_pkl}')
    else:
        print(f'Selecting {args.num_scenes} random scenes per dataset with {args.num_workers} workers')
    
    if args.dataset == 'all' or args.dataset == 'training':
        process('training', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir, args.num_workers, args.scene_ids_pkl)

    if args.dataset == 'all' or args.dataset == 'testing':
        process('testing', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir, args.num_workers, args.scene_ids_pkl)
            
    if args.dataset == 'all' or args.dataset == 'validation':
        process('validation', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir, args.num_workers, args.scene_ids_pkl)
        
    if args.dataset == 'all' or args.dataset == 'validation_interactive':
        process('validation_interactive', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir, args.num_workers, args.scene_ids_pkl)