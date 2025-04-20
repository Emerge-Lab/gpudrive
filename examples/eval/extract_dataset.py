import os
# Force JAX to use CPU only
# os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
import glob
import argparse
# import pickle
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2


# from waymax import dataloader
# from waymax.config import DataFormat
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
    """
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type=""
    )
       
    os.makedirs(save_dir, exist_ok=True)
    
    for tf_data in dataset:
        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))
        scenario_id = scenario.scenario_id
                
        scenario_filename = os.path.join(save_dir, +scenario_id+'.tfrecords')
        
        # check if file exists
        if os.path.exists(scenario_filename):
            continue
        
        with tf.io.TFRecordWriter(scenario_filename.as_posix()) as file_writer:
            file_writer.write(tf_data)


if __name__ == '__main__': 
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/Dataset/Waymo/V1_2_tf')
    parser.add_argument('--save_dir', type=str, default='/data/Dataset/Waymo/VBD')
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--num_workers', type=int, default=1)  # Change default to 1
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f'Processing data from {args.data_dir} and Saving to {args.save_dir}')
    
    def process(dataset):
        """
        Process a specific dataset and save the processed data.

        Args:
            dataset (str): Name of the dataset to process.
        """
        data_path = os.path.join(args.data_dir, dataset+'/*')
        data_files = glob.glob(data_path)
        save_dir = os.path.join(args.save_dir, dataset+'_extracted')
        n_files = len(data_files)
        print(f'Processing {n_files} files in {data_path} dataset')
        os.makedirs(save_dir, exist_ok=True)
        print(f'Saving to {save_dir}')
        
        # Process files one by one instead of using multiprocessing
        for data_file in tqdm(data_files):
            data_process(data_file, save_dir)
        
            
    if args.dataset == 'all' or args.dataset == 'train':
        process('training')
            
    if args.dataset == 'all' or args.dataset == 'val':
        process('validation')
        
    if args.dataset == 'all' or args.dataset == 'val_interactive':
        process('validation_interactive')