import os
# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
import glob
import argparse
import random
import json
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from waymax import dataloader
from waymax.config import DataFormat

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

def process(dataset, num_scenes, data_dir, tfrecord_dir, save_dir):
    """
    Process a specific dataset, selecting random scenes and extracting their TFRecords.
    
    Args:
        dataset (str): Name of the dataset to process
        num_scenes (int): Number of scenes to randomly select
        data_dir (str): Directory containing JSON files
        tfrecord_dir (str): Directory containing TFRecord files
        save_dir (str): Directory to save processed files
    """
    # Set up paths
    data_path = os.path.join(data_dir, dataset, '*.json')
    json_files = glob.glob(data_path)
    
    # Create output directories
    json_save_dir = os.path.join(save_dir, f"{dataset}_json")
    tfrecord_save_dir = os.path.join(save_dir, f"{dataset}_tfrecord")
    os.makedirs(json_save_dir, exist_ok=True)
    os.makedirs(tfrecord_save_dir, exist_ok=True)
    
    # Randomly select scenes
    n_files = len(json_files)
    if n_files == 0:
        print(f"No JSON files found in {data_path}")
        return
    
    print(f"Found {n_files} files in {dataset} dataset")
    num_to_select = min(num_scenes, n_files)
    selected_files = random.sample(json_files, num_to_select)
    
    print(f"Processing {num_to_select} randomly selected files from {dataset}")
    
    for json_file in tqdm(selected_files):
        # Extract prefix and suffix from filename
        filename = os.path.basename(json_file)
        name_parts = os.path.splitext(filename)[0].split('_', 1)  # Split on the first underscore only
        if len(name_parts) != 2:
            print(f"Invalid filename format: {filename}, expected <prefix>_<suffix>.json")
            continue
        prefix, suffix = name_parts
        
        # Find corresponding TFRecord file
        tfrecord_path = os.path.join(tfrecord_dir, dataset, f"{dataset}_tfexample.{prefix}")
        
        if not os.path.exists(tfrecord_path):
            print(f"TFRecord file not found: {tfrecord_path}")
            continue
        
        # Copy the JSON file to the output directory with new name
        json_output_path = os.path.join(json_save_dir, f"{suffix}.json")
        
        # Copy the JSON file content
        with open(json_file, 'r') as f_in:
            json_data = json.load(f_in)
        
        with open(json_output_path, 'w') as f_out:
            json.dump(json_data, f_out)
        
        # Extract the scene from the TFRecord file
        tfrecord_output_path = os.path.join(tfrecord_save_dir, f"{suffix}.tfrecord")
        found = extract_scene_from_tfrecord(tfrecord_path, suffix, tfrecord_output_path)
        
        if not found:
            print(f"Scene {suffix} not found in TFRecord file: {tfrecord_path}")
            # Remove the JSON file if the TFRecord extraction failed
            os.remove(json_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/processed/')
    parser.add_argument('--save_dir', type=str, default='/data/processed/wosac/')
    parser.add_argument('--tfrecord_dir', type=str, default='/data/raw/')
    parser.add_argument('--dataset', type=str, default='validation')
    parser.add_argument('--num_scenes', type=int, default=1000)  # New parameter for selecting n scenes
    parser.add_argument('--num_workers', type=int, default=1)  # Kept from original code
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f'Processing data from {args.data_dir} and Saving to {args.save_dir}')
    print(f'Selecting {args.num_scenes} random scenes per dataset')
    
    if args.dataset == 'all' or args.dataset == 'training':
        process('training', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir)

    if args.dataset == 'all' or args.dataset == 'testing':
        process('testing', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir)
            
    if args.dataset == 'all' or args.dataset == 'validation':
        process('validation', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir)
        
    if args.dataset == 'all' or args.dataset == 'validation_interactive':
        process('validation_interactive', args.num_scenes, args.data_dir, args.tfrecord_dir, args.save_dir)