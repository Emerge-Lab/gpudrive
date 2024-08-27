import copy
import numpy as np
import tensorflow as tf

from waymax import datatypes
from waymax import dataloader
from waymax import config as waymax_config
from typing import Dict, Tuple, List
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def create_iter(data_config: waymax_config.DatasetConfig)-> iter(Tuple[str, datatypes.SimulatorState]):
    # Write a custom dataloader that loads scenario IDs.
    def _preprocess(serialized: bytes) -> dict[str, tf.Tensor]:
        womd_features = dataloader.womd_utils.get_features_description(
            include_sdc_paths=data_config.include_sdc_paths,
            max_num_rg_points=data_config.max_num_rg_points,
            num_paths=data_config.num_paths,
            num_points_per_path=data_config.num_points_per_path,
        )
        womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

        deserialized = tf.io.parse_example(serialized, womd_features)
        parsed_id = deserialized.pop('scenario/id')
        deserialized['scenario/id'] = tf.io.decode_raw(parsed_id, tf.uint8)
        return dataloader.preprocess_womd_example(
            deserialized,
            aggregate_timesteps=data_config.aggregate_timesteps,
            max_num_objects=data_config.max_num_objects,
        )
        
    def _postprocess(example: dict[str, tf.Tensor]):
        scenario = dataloader.simulator_state_from_womd_dict(example)
        scenario_id = example['scenario/id']
        return scenario_id, scenario
    
    def decode_bytes(data_iter):
        # Force use CPU
        # with tf.device('/cpu:0'):
        for scenario_id, scenario in data_iter:
            scenario_id = scenario_id.tobytes().decode('utf-8')
            yield scenario_id, scenario
                
    return decode_bytes(dataloader.get_data_generator(
            data_config, _preprocess, _postprocess
        ))
        
        
class WomdLoader:
    def __init__(self, data_config: waymax_config.DatasetConfig) -> None:
        self.data_config = data_config
        self.reset()
        
    def reset(self):
        self.iter = create_iter(self.data_config)
    
    def next(self):
        return next(self.iter)
    

def smooth_scenario(scenario: datatypes.SimulatorState, window_size=11, polyorder=3, duplicate=False):
    """
    Smooths the trajectory of a scenario by applying filtering and interpolation techniques.

    Args:
        scenario (datatypes.SimulatorState): The scenario to be smoothed.
        window_size (int, optional): The size of the window used for smoothing. Defaults to 11.
        polyorder (int, optional): The order of the polynomial used for smoothing. Defaults to 3.

    Returns:
        the updated scenario.
    """
    if duplicate:
        scenario = copy.deepcopy(scenario)
    
    traj = scenario.log_trajectory
    original_valid = np.asarray(traj.valid)
    vel = np.stack(
        [traj.vel_x, traj.vel_y, np.sin(traj.yaw), np.cos(traj.yaw)], axis=-1
    )

    num_agent, num_step = traj.valid.shape
    smoothed_vel_x = np.zeros_like(traj.vel_x)
    smoothed_vel_y = np.zeros_like(traj.vel_y)
    smoothed_yaw = np.zeros_like(traj.yaw)
    smoothed_valid = np.zeros_like(traj.valid, dtype=bool)

    t = np.arange(num_step)

    for i in range(num_agent):
        # Extract raw data and valid mask
        valid = original_valid[i]
        t_valid = t[valid]
        vel_valid = vel[i][valid, :]
        valid_idx = np.where(valid)[0]
            
        if len(valid_idx) == 0: # skip if no valid data
            continue
        
        # Use zscore to filter out outliers
        std = np.clip(np.std(vel_valid, axis=-2, keepdims=True), a_min = 0.1, a_max=None)
        mean = np.mean(vel_valid, axis=-2, keepdims=True)
        z = np.abs((vel_valid-mean)/std)
        filtered_idx = np.all(z < 4, axis=-1)
        valid_idx = valid_idx[filtered_idx]
        
        
        if len(valid_idx) == 0: # skip if no valid data
            continue

        first_valid_idx = valid_idx[0]
        last_valid_idx = valid_idx[-1]
        if (last_valid_idx - first_valid_idx) <= 3:
            continue
        
        # Extract valid velocity data and interpolate
        t_valid = t[valid_idx]

        vel_valid = vel[i][valid_idx, :]
        vel_interp = interp1d(t_valid, vel_valid, axis=0, kind='linear')

        t_interped = np.arange(first_valid_idx, last_valid_idx+1)
        vel_interped = vel_interp(t_interped)
        
        # Smooth the interpolated data
        vel_smoothed = savgol_filter(vel_interped,
                        min(last_valid_idx-first_valid_idx, window_size),
                        polyorder,
                        axis=0
                    )
        
        # update smoothed velocity
        smoothed_vel_x[i, first_valid_idx:last_valid_idx+1] = vel_smoothed[:, 0]
        smoothed_vel_y[i, first_valid_idx:last_valid_idx+1] = vel_smoothed[:, 1]
        smoothed_yaw[i, first_valid_idx:last_valid_idx+1] = np.arctan2(vel_smoothed[:, 2], vel_smoothed[:, 3])
        smoothed_valid[i, first_valid_idx:last_valid_idx+1] = True
    
    # Update Scenario
    scenario.log_trajectory.vel_x = smoothed_vel_x
    scenario.log_trajectory.vel_y = smoothed_vel_y
    scenario.log_trajectory.yaw = smoothed_yaw
    scenario.log_trajectory.valid = np.logical_and(original_valid, smoothed_valid)

    return scenario