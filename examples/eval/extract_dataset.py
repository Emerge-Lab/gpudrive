import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# set tf to cpu only
tf.config.set_visible_devices([], 'GPU')

import glob
import argparse
import pickle
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

from waymax import dataloader
from waymax.config import DataFormat
import functools

MAX_NUM_OBJECTS = 64
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30

def tf_preprocess(serialized: bytes) -> dict[str, tf.Tensor]:
    """
    Preprocesses the serialized data.

    Args:
        serialized (bytes): The serialized data.

    Returns:
        dict[str, tf.Tensor]: The preprocessed data.
    """
    womd_features = dataloader.womd_utils.get_features_description(
        include_sdc_paths=False,
        max_num_rg_points=30000,
        num_paths=None,
        num_points_per_path=None,
    )
    womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

    deserialized = tf.io.parse_example(serialized, womd_features)
    parsed_id = deserialized.pop('scenario/id')
    deserialized['scenario/id'] = tf.io.decode_raw(parsed_id, tf.uint8)
    return dataloader.preprocess_womd_example(
        deserialized,
        aggregate_timesteps=True,
        max_num_objects=None,
    )

def tf_postprocess(example: dict[str, tf.Tensor]):
    """
    Postprocesses the example.

    Args:
        example (dict[str, tf.Tensor]): The example to be postprocessed.

    Returns:
        tuple: A tuple containing the scenario ID and the postprocessed scenario.
    """
    scenario = dataloader.simulator_state_from_womd_dict(example)
    scenario_id = example['scenario/id']
    return scenario_id, scenario

def data_process(
    data_dir: str, 
    save_dir: str, 
):
    """
    Process the Waymax dataset and save the processed data.

    Args:
        data_dir (str): Directory path of the Waymax dataset.
        save_dir (str): Directory path to save the processed data.
        save_raw (bool, optional): Whether to save the raw scenario data. Defaults to False.
    """
    # Waymax Dataset
    tf_dataset = dataloader.tf_examples_dataset(
        path=data_dir,
        data_format=DataFormat.TFRECORD,
        preprocess_fn=tf_preprocess,
        repeat=1,
        # num_shards=16,
        deterministic=True,
    )
    
    tf_dataset_iter = tf_dataset.as_numpy_iterator()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for example in tf_dataset_iter:
        
        scenario_id_binary, scenario = tf_postprocess(example)
        scenario_id = scenario_id_binary.tobytes().decode('utf-8')
        
        scenario_filename = os.path.join(save_dir, 'scenario_'+scenario_id+'.pkl')
        
        # check if file exists
        if os.path.exists(scenario_filename):
            continue
        
        data_dict = {'scenario_raw': scenario}
        data_dict['scenario_id'] = scenario_id
        

        with open(scenario_filename, 'wb') as f:
            pickle.dump(data_dict, f)
        

if __name__ == '__main__': 
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/Dataset/Waymo/V1_2_tf')
    parser.add_argument('--save_dir', type=str, default='/data/Dataset/Waymo/VBD')
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f'Processing data from {args.data_dir} and Saving to {args.save_dir}')
    
    def process(dataset):
        """
        Process a specific dataset and save the processed data.

        Args:
            dataset (str): Name of the dataset to process.
            save_raw (bool, optional): Whether to save the raw scenario data. Defaults to False.
        """
        data_path = os.path.join(args.data_dir, dataset+'/*')
        data_files = glob.glob(data_path)
        save_dir = os.path.join(args.save_dir, dataset+'_extracted')
        n_files = len(data_files)
        print(f'Processing {n_files} files in {data_path} dataset')
        os.makedirs(save_dir, exist_ok=True)
        print(f'Saving to {save_dir}')
        data_process_partial = functools.partial(
            data_process, 
            save_dir=save_dir,
        )
        _ = process_map(data_process_partial, data_files, max_workers=args.num_workers)
        
            
    if args.dataset == 'all' or args.dataset == 'train':
        process('training', save_raw=args.save_raw, only_raw=args.only_raw)
            
    if args.dataset == 'all' or args.dataset == 'val':
        process('validation', save_raw=args.save_raw, only_raw=args.only_raw)
        
    if args.dataset == 'all' or args.dataset == 'val_interactive':
        process('validation_interactive', save_raw=args.save_raw, only_raw=args.only_raw)
    
    
            
