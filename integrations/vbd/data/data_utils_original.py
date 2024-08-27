import tensorflow as tf
import torch
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from waymax import dataloader
from waymax import datatypes

def plot_scenario(agents, futures, polyilnes, traffic_lights):
    """
    Plot the scenario with agents, futures, polylines, and traffic lights.

    Args:
        agents (numpy.ndarray): Array of agent positions and orientations.
        futures (numpy.ndarray): Array of future trajectories.
        polyilnes (numpy.ndarray): Array of polyline positions.
        traffic_lights (numpy.ndarray): Array of traffic light positions.
    """
    # plot map
    for i in range(polyilnes.shape[0]):
        if polyilnes[i, 0, 0] != 0:
            line = polyilnes[i]
            line = line[line[:, 0] != 0]
            plt.plot(line[:, 0], line[:, 1], '.', markersize=2)

    # plot agents
    for i in range(agents.shape[0]):
        pos_x, pos_y = agents[i, -1, 0], agents[i, -1, 1]
        heading = agents[i, -1, 2]
        length, width = agents[i, -1, 5], agents[i, -1, 6]
        rect = plt.Rectangle(
            (pos_x - length/2, pos_y - width/2), length, width, linewidth=2, color='b', alpha=0.9, zorder=3,
            transform=mpl.transforms.Affine2D().rotate_around(*(pos_x, pos_y), heading) + plt.gca().transData
        )
        plt.gca().add_patch(rect)
        if agents[i, -1, 0] != 0:
            agent = agents[i][agents[i, :, 0] != 0]
            plt.plot(agent[:, 0], agent[:, 1], 'b')

    # plot traffic lights
    for i in range(traffic_lights.shape[0]):
        if traffic_lights[i, 0] != 0:
            circle = plt.Circle((traffic_lights[i, 0], traffic_lights[i, 1]), 1, color='k', alpha=0.9, zorder=3)
            plt.gca().add_patch(circle)

    # plot futures
    for i in range(futures.shape[0]):
        if futures[i, 0, 0] != -1:
            line = futures[i][futures[i, :, 0] != -1]
            plt.plot(line[:, 0], line[:, 1], 'r')
        
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        angle (float): The input angle.

    Returns:
        float: The wrapped angle.
    """
    # return (angle + np.pi) % (2 * np.pi) - np.pi
    return np.arctan2(np.sin(angle), np.cos(angle))

def filter_topk_roadgraph_points(roadgraph, reference_points, topk):
    """
    Returns the topk closest roadgraph points to a reference point.

    If `topk` is larger than the number of points, an exception will be raised.

    Args:
        roadgraph: Roadgraph information to filter, (..., num_points).
        reference_points: A tensor of shape (..., 2) - the reference point used to measure distance.
        topk: Number of points to keep.

    Returns:
        Roadgraph data structure that has been filtered to only contain the `topk` closest points to a reference point.
    """
    if topk > roadgraph.num_points:
        raise NotImplementedError('Not enough points in roadgraph.')
    
    elif topk < roadgraph.num_points:
        roadgraph_xy = np.asarray(roadgraph.xy)
        distances = np.linalg.norm(reference_points[..., None, :] - roadgraph_xy, axis=-1)
        valid_distances = np.where(roadgraph.valid, distances, float('inf'))
        top_idx = np.argpartition(valid_distances, topk, axis=-1)[..., :topk]

        stacked = np.stack(
            [
                roadgraph.x,
                roadgraph.y,
                roadgraph.z,
                roadgraph.dir_x,
                roadgraph.dir_y,
                roadgraph.dir_z,
                roadgraph.types,
                roadgraph.ids,
                roadgraph.valid,
            ],
            axis=-1,
        )
        filtered = np.take_along_axis(stacked, top_idx[..., None], axis=-2)

        return datatypes.RoadgraphPoints(
            x=filtered[..., 0],
            y=filtered[..., 1],
            z=filtered[..., 2],
            dir_x=filtered[..., 3],
            dir_y=filtered[..., 4],
            dir_z=filtered[..., 5],
            types=filtered[..., 6].astype(np.int32),
            ids=filtered[..., 7].astype(np.int32),
            valid=filtered[..., 8].astype(np.bool_),
        )
    
    else:
        return roadgraph
    
def calculate_relations(agents, polylines, traffic_lights):
    """
    Calculate the relations between agents, polylines, and traffic lights.

    Args:
        agents (numpy.ndarray): Array of agent positions and orientations.
        polylines (numpy.ndarray): Array of polyline positions.
        traffic_lights (numpy.ndarray): Array of traffic light positions.

    Returns:
        numpy.ndarray: Array of relations between the elements.
    """
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights

    # Prepare a single array to hold all elements
    all_elements = np.concatenate([
        agents[:, -1, :3],
        polylines[:, 0, :3],
        np.concatenate([traffic_lights[:, :2], np.zeros((n_traffic_lights, 1))], axis=1)
    ], axis=0)

    # Compute pairwise differences using broadcasting
    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]

    # Compute local positions and angle differences
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])

    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    theta_diff = np.where((np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :], 0, theta_diff)

    # Set the diagonal of the differences to a very small value
    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    # Conditions for zero coordinates
    zero_mask = np.logical_or(all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0)

    # Initialize relations array
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)

    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)

    return relations

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

def data_process_agent(
    scenario,
    current_index: int = 10,
    max_num_objects: int = 64,
    history_length: int = 11,
    use_log: bool = True,
    selected_agents: bool = None,
):
    """
    Process the data for surrounding agents in a given scenario.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing the agent data.
        max_num_objects (int): The maximum number of objects to consider.
        current_index (int): The current time index.
        use_log (bool): Whether to use log trajectory or sim trajectory.
        selected_agents (list[int] or None): List of agent IDs to consider. If None, all agents will be considered.

    Returns:
        tuple: A tuple containing the processed agent data, including:
            - agents_history (ndarray): The history of agent trajectories. Shape: (max_object, history_length, 8)
            - agents_future (ndarray): The future agent trajectories. Shape: (max_object, future_length, 5)
            - agents_interested (ndarray): The interest level of agents. Shape: (max_object,)
            - agents_type (ndarray): The type of agents. Shape: (max_object,)
    """
    if use_log:
        log_trajectory = scenario.log_trajectory
    else:
        log_trajectory = scenario.sim_trajectory
        
    metadata = scenario.object_metadata

    # # calculate distance to sdc
    # if selected_agents is None:
    #     sdc_id = np.where(metadata.is_sdc)[0][0]
    #     sdc_position = np.asarray(log_trajectory.xy[sdc_id, current_index])
    #     agents_positions = np.asarray(log_trajectory.xy[:, current_index])
    #     distance_to_sdc = np.linalg.norm(agents_positions - sdc_position, axis=-1)
    #     agent_ids = np.argsort(distance_to_sdc)[:max_num_objects]
    #     # sort agent_ids
    #     agent_ids = np.sort(agent_ids)
    # else:
    #     agent_ids = selected_agents
    
    # Top k agents
    agent_ids = list(range(max_num_objects))
    
    ############# Get agents' trajectory #############
    # feature: x, y, yaw, velx, vely, length, width, height
    agents_valid = np.asarray(log_trajectory.valid)
    agents_valid = np.take(agents_valid, agent_ids, axis=0)
    # pad to max_num_objects
    agents_valid = np.pad(agents_valid, ((0, max_num_objects-agents_valid.shape[0]), (0,0)), mode='constant', constant_values=False)
        
    agents_traj = np.stack([
        log_trajectory.x,
        log_trajectory.y,
        log_trajectory.yaw,
        log_trajectory.vel_x,
        log_trajectory.vel_y,
        log_trajectory.length,
        log_trajectory.width,
        log_trajectory.height,
    ], axis=-1)
    agents_traj = np.take(agents_traj, agent_ids, axis=0)# [max_object, N step, 8]
    agents_traj = np.pad(agents_traj, ((0, max_num_objects-agents_traj.shape[0]),(0, 0),(0, 0)), mode='constant', constant_values=0.0)
    
    
    # Filter out invalid agents
    agents_traj = np.where(
        agents_valid[..., None],
        agents_traj,
        0.0
    )
    
    # Some trajectories having zero in data but marked as valid
    agents_valid_data = np.all(agents_traj[..., :2]!=0, axis=-1)
    agents_valid = np.logical_and(agents_valid, agents_valid_data)
    
    agents_history_valid = agents_valid[:, current_index-history_length+1:current_index+1]
    agents_history = agents_traj[:, current_index-history_length+1:current_index+1] # [max_object, history_length, 8]
    
    future_length = agents_traj.shape[1] - history_length + 1
    agents_future = agents_traj[:, current_index:] # [max_object, future_length, 8]
    agents_future_valid = agents_valid[:, current_index:] # [max_object, future_length]

    agents_future = np.pad(agents_future, 
            ((0, 0), (0, future_length - agents_future.shape[1]), (0, 0)), 
            mode='constant', constant_values=0.0)
    agents_future_valid = np.pad(agents_future_valid,
            ((0, 0), (0, future_length - agents_future_valid.shape[1])),
            mode='constant', constant_values=False)

    agents_type = np.asarray(metadata.object_types)
    agents_type = np.take(agents_type, agent_ids, axis=0)
    agents_type = np.pad(agents_type, (0, max_num_objects-agents_type.shape[0]), mode='constant', constant_values=0)
    agents_type = np.where(
        agents_type>0,
        agents_type,
        0
    )
    
    is_modeled = np.take(np.asarray(metadata.is_modeled), agent_ids, axis=0)
    is_modeled = np.pad(is_modeled, (0, max_num_objects-is_modeled.shape[0]), mode='constant', constant_values=False)
    objects_of_interest = np.take(np.asarray(metadata.objects_of_interest), agent_ids, axis=0)
    objects_of_interest = np.pad(objects_of_interest, (0, max_num_objects-objects_of_interest.shape[0]), mode='constant', constant_values=False)

    agents_interested = np.int32(agents_history_valid[:, -1])
    agents_interested = np.where(
        (is_modeled | objects_of_interest),
        10,
        agents_interested
    )
    
    # Padding agent_ids
    agent_ids = np.pad(agent_ids, (0, max_num_objects-len(agent_ids)), mode='constant', constant_values=-1)

    return (agents_history, agents_history_valid,
            agents_future, agents_future_valid,
            agents_interested, agents_type,
            agent_ids,
        )

def data_process_traffic_light(
    scenario: datatypes.SimulatorState,
    current_index: int = 10
):
    """
    Process traffic light data from the given scenario.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing traffic light information.
        current_index (int, optional): The index of the current time step. Defaults to 10.

    Returns:
        tuple: A tuple containing the processed traffic light points, lane IDs, and states.
    """
    traffic_lights = scenario.log_traffic_light
    ############# Get Traffic Lights #############
    traffic_lane_ids = np.asarray(traffic_lights.lane_ids)[:, current_index]
    traffic_light_states = np.asarray(traffic_lights.state)[:, current_index]
    traffic_stop_points = np.asarray(traffic_lights.xy)[:, current_index]
    traffic_light_valid = np.asarray(traffic_lights.valid)[:, current_index]
    
    traffic_light_points = np.concatenate([traffic_stop_points, traffic_light_states[:, None]], axis=1)    
    traffic_light_points = np.float32(traffic_light_points)
    traffic_light_points = np.where(
        traffic_light_valid[:, None],
        traffic_light_points,
        0.0
    )
    
    return traffic_light_points, traffic_lane_ids, traffic_light_states

def data_process_scenario(
    scenario: datatypes.SimulatorState,
    max_num_objects: int = 64,
    max_polylines: int = 256,
    current_index: int = 10,
    num_points_polyline: int = 30,
    use_log: bool = True,
    selected_agents = None,
):
    """
    Process the data for a given scenario.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing the scenario data.

    Returns:
        dict: A dictionary containing the processed data.
    """
    output_dict = {} 
    
    (agents_history, agents_history_valid, agents_future, 
        agents_future_valid, agents_interested,
        agents_type, agent_ids) = data_process_agent(
            scenario,
            max_num_objects = max_num_objects,
            current_index = current_index,
            use_log = use_log,
            selected_agents = selected_agents,
        )
                    
    (traffic_light_points, traffic_lane_ids, traffic_light_states) = data_process_traffic_light(
        scenario,
        current_index = current_index,
    )
    
    roadgraph_points = scenario.roadgraph_points
    
    ############### get roadgraph points near agents ###############
    map_ids = []
    current_valid = agents_history_valid[:, -1]
    current_positions = np.asarray(scenario.log_trajectory.xy)[:, current_index]
    
    for a in range(agents_history.shape[0]):
        if not current_valid[a]:
            continue
        
        agent_position = current_positions[a]
        nearby_roadgraph_points = filter_topk_roadgraph_points(roadgraph_points, agent_position, 3000)
        map_ids.append(nearby_roadgraph_points.ids.tolist())

    # sort map ids
    sorted_map_ids = []
    for i in range(nearby_roadgraph_points.shape[0]):
        for j in range(len(map_ids)):
            if map_ids[j][i] != -1 and map_ids[j][i] not in sorted_map_ids:
                sorted_map_ids.append(map_ids[j][i])
    
    # get shared map polylines
    # polyline feature: x, y, heading, traffic_light, type
    polylines = []
    
    roadgraph_points_x = np.asarray(roadgraph_points.x)
    roadgraph_points_y = np.asarray(roadgraph_points.y)
    roadgraph_points_dir_x = np.asarray(roadgraph_points.dir_x)
    roadgraph_points_dir_y = np.asarray(roadgraph_points.dir_y)
    roadgraph_points_types = np.asarray(roadgraph_points.types)

    for id in sorted_map_ids:
        # get polyline
        p_x = roadgraph_points_x[roadgraph_points.ids == id]
        p_y = roadgraph_points_y[roadgraph_points.ids == id]
        dir_x = roadgraph_points_dir_x[roadgraph_points.ids == id]
        dir_y = roadgraph_points_dir_y[roadgraph_points.ids == id]
        heading = np.arctan2(dir_y, dir_x)
        lane_type = roadgraph_points_types[roadgraph_points.ids == id]
        traffic_light_state = traffic_light_states[traffic_lane_ids == id] if id in traffic_lane_ids else 0
        traffic_light_state = np.repeat(traffic_light_state, len(p_x))
        polyline = np.stack([p_x, p_y, heading, traffic_light_state, lane_type], axis=1)
        
        # sample points and fill into fixed-size array
        polyline_len = polyline.shape[0]
        sampled_points = np.linspace(0, polyline_len-1, num_points_polyline, dtype=np.int32)
        cur_polyline = np.take(polyline, sampled_points, axis=0)
        polylines.append(cur_polyline)
    
    # post processing polylines
    if len(polylines) > 0:
        polylines = np.stack(polylines, axis=0)
        polylines_valid = np.ones((polylines.shape[0],), dtype=np.int32)
    else:
        polylines = np.zeros((1, num_points_polyline, 5), dtype=np.float32)
        polylines_valid = np.zeros((1,), dtype=np.int32)
    
    if polylines.shape[0] >= max_polylines:
        polylines = polylines[:max_polylines]
        polylines_valid = polylines_valid[:max_polylines]
    else:
        polylines = np.pad(polylines, ((0, max_polylines-polylines.shape[0]), (0, 0), (0, 0)))
        polylines_valid = np.pad(polylines_valid, (0, max_polylines-polylines_valid.shape[0]))

    relations = calculate_relations(agents_history, polylines, traffic_light_points)
    relations = np.asarray(relations)
    
    output_dict.update(
        {
            'agents_history': np.float32(agents_history),
            'agents_interested': np.int32(agents_interested),
            'agents_type': np.int32(agents_type),
            'agents_future': np.float32(agents_future), 
            'agents_future_valid': np.bool_(agents_future_valid),
            'traffic_light_points': np.float32(traffic_light_points),
            'polylines': np.float32(polylines),
            'polylines_valid': np.int32(polylines_valid),
            'relations': np.float32(relations),
            'agent_ids': np.int32(agent_ids),
        }
    )
    return output_dict

def data_collate_fn(batch_list):
    """
    Collects a batch of data from a list of transitions.

    Args:
        batch_list (List): a list of transitions.

    Returns:
        Dict[str, torch.Tensor]: a batch of data.
    """
    list_len = len(batch_list)
    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(list_len)]
        
    input_batch = {}
    for key, value in key_to_list.items():
        if 'scenario' not in key:
            input_batch[key] = torch.from_numpy(np.stack(value, axis=0))
        else:
            input_batch[key] = value
            
    return input_batch