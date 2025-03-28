"""Functions used for processing roadgraph data and other features for VBD."""
import torch
import numpy as np
import madrona_gpudrive
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints

def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        angle (float): The input angle.

    Returns:
        float: The wrapped angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def filter_topk_roadgraph_points(global_road_graph, reference_points, topk):
    """
    Returns the topk closest roadgraph points to a reference point.

    If `topk` is larger than the number of points, an exception will be raised.

    Args:
        roadgraph: Roadgraph information to filter, GlobalRoadGraphPoints.
        reference_points: A tensor of shape (..., 2) - the reference point used to measure distance.
        topk: Number of points to keep.

    Returns:
        GlobalRoadGraphPoints data structure that has been filtered to only contain the `topk` closest points to a reference point.
    """
    if topk > global_road_graph.num_points:
        raise NotImplementedError("Not enough points in roadgraph.")

    elif topk < global_road_graph.num_points:
        roadgraph_xy = np.asarray(global_road_graph.xy[0])
        distances = np.linalg.norm(
            reference_points[..., None, :] - roadgraph_xy, axis=-1
        )
        valid_distances = np.where(global_road_graph.id > 0, distances, float("inf"))
        top_idx = np.argpartition(valid_distances, topk, axis=-1)[..., :topk]

        # Gather the topk points by slicing along the indices
        filtered_xy = global_road_graph.xy[0][top_idx]
        filtered_length = global_road_graph.segment_length[0][top_idx]
        filtered_width = global_road_graph.segment_width[0][top_idx]
        filtered_height = global_road_graph.segment_height[0][top_idx]
        filtered_orientation = global_road_graph.orientation[0][top_idx]
        filtered_type = global_road_graph.vbd_type[0][top_idx]
        filtered_id = global_road_graph.id[0][top_idx]

        # Stack the filtered attributes to form a new roadgraph tensor
        filtered_tensor = torch.stack(
            [
                filtered_xy[..., 0],
                filtered_xy[..., 1],
                filtered_length,
                filtered_width,
                filtered_height,
                filtered_orientation,
                torch.zeros_like(filtered_length),
                filtered_id,
                filtered_type
            ],
            dim=-1
        )

        return GlobalRoadGraphPoints(filtered_tensor.clone())
    else:
        return global_road_graph


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
    all_elements = np.concatenate(
        [
            agents[:, -1, :3],
            polylines[:, 0, :3],
            np.concatenate(
                [traffic_lights[:, :2], np.zeros((n_traffic_lights, 1))],
                axis=1,
            ),
        ],
        axis=0,
    )

    # Compute pairwise differences using broadcasting
    pos_diff = (
        all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]
    )

    # Compute local positions and angle differences
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(
        all_elements[:, 2][:, None] - all_elements[:, 2][None, :]
    )

    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    theta_diff = np.where(
        (np.arange(n) >= start_idx)[:, None]
        | (np.arange(n) >= start_idx)[None, :],
        0,
        theta_diff,
    )

    # Set the diagonal of the differences to a very small value
    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    # Conditions for zero coordinates
    zero_mask = np.logical_or(
        all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0
    )

    # Initialize relations array
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)

    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)

    return relations


def process_agents_vectorized(num_worlds, max_cont_agents, init_steps, global_agent_obs, log_trajectory, metadata, raw_agent_types, controlled_agent_mask):
    """
    Vectorized function to process agent data across multiple worlds.
    Using controlled_agent_mask instead of SDC proximity.
    """
    # Initialize output arrays with batch dimension
    agents_history = np.zeros((num_worlds, max_cont_agents, init_steps + 1, 8), dtype=np.float32)
    agents_type = np.zeros((num_worlds, max_cont_agents), dtype=np.int32)
    agents_interested = np.zeros((num_worlds, max_cont_agents), dtype=np.int32)
    agents_future = np.zeros(
        (num_worlds, max_cont_agents, log_trajectory.pos_xy.shape[2] - init_steps, 5),
        dtype=np.float32
    )
    agents_id = np.zeros((num_worlds, max_cont_agents), dtype=np.int32)
    
    # Process each world using controlled_agent_mask
    for w in range(num_worlds):
        # Get indices of controlled agents
        controlled_indices = np.where(controlled_agent_mask[w])[0]
        
        # Sort by agent ID for consistency
        sorted_agent_indices = np.sort(controlled_indices)

        # Handle case where we have fewer controlled agents than max_cont_agents
        if len(sorted_agent_indices) < max_cont_agents:
            # Pad with -1 i.e. invalid agent index
            padded_indices = np.full(max_cont_agents, -1, dtype=np.int32)
            padded_indices[:len(sorted_agent_indices)] = sorted_agent_indices
            sorted_agent_indices = padded_indices

        # Store agent indices
        agents_id[w] = sorted_agent_indices
        
        # Process each agent for this world
        for i, a in enumerate(sorted_agent_indices):
            if a == -1:
                break
            agent_type = raw_agent_types[w][a] if isinstance(raw_agent_types, list) else raw_agent_types[w, a]
            valid = log_trajectory.valids[w, a, init_steps]
            
            if valid.item() != 1:
                agents_interested[w, i] = 0
                continue
                
            if metadata.isModeled[w, a] or metadata.isOfInterest[w, a]:
                agents_interested[w, i] = 10
            else:
                agents_interested[w, i] = 1
            
            agents_type[w, i] = agent_type
            agents_history[w, i] = torch.column_stack(
                [
                    log_trajectory.pos_xy[w, a, :init_steps+1, 0],
                    log_trajectory.pos_xy[w, a, :init_steps+1, 1],
                    log_trajectory.yaw[w, a, :init_steps+1, 0],
                    log_trajectory.vel_xy[w, a, :init_steps+1, 0],
                    log_trajectory.vel_xy[w, a, :init_steps+1, 1],
                    global_agent_obs.vehicle_length[w, a].repeat(init_steps + 1),
                    global_agent_obs.vehicle_width[w, a].repeat(init_steps + 1),
                    global_agent_obs.vehicle_height[w, a].repeat(init_steps + 1),
                ],
            ).numpy()
            
            mask = log_trajectory.valids[w, a, :init_steps+1].numpy()
            agents_history[w, i] *= mask
            
            agents_future[w, i] = torch.column_stack(
                [
                    log_trajectory.pos_xy[w, a, init_steps:, 0],
                    log_trajectory.pos_xy[w, a, init_steps:, 1],
                    log_trajectory.yaw[w, a, init_steps:, 0],
                    log_trajectory.vel_xy[w, a, init_steps:, 0],
                    log_trajectory.vel_xy[w, a, init_steps:, 1],
                ],
            ).numpy()
            
            mask = log_trajectory.valids[w, a, init_steps:].numpy()
            agents_future[w, i] *= mask
    
    # Map agent types for all worlds at once (this is vectorized)
    mapped_agents_type = np.zeros_like(agents_type)
    mapped_agents_type[agents_type == int(madrona_gpudrive.EntityType.Vehicle)] = 1
    mapped_agents_type[agents_type == int(madrona_gpudrive.EntityType.Pedestrian)] = 2
    mapped_agents_type[agents_type == int(madrona_gpudrive.EntityType.Cyclist)] = 3
    
    return agents_history, agents_future, agents_interested, mapped_agents_type, agents_id

def process_world_roadgraph(global_road_graph, world_idx, agents_history, agents_interested, max_polylines, num_points_polyline):
    """
    Process the roadgraph for a single world.
    """
    # Extract the world's roadgraph data
    world_road_graph = extract_world_data(global_road_graph, world_idx)
    
    # Get map IDs based on agent positions
    map_ids = []
    current_valid = agents_interested > 0
    
    for agent_idx in range(agents_history.shape[0]):
        if not current_valid[agent_idx]:
            continue
        agent_position = agents_history[agent_idx, -1, :2]
        nearby_roadgraph_points = filter_topk_roadgraph_points(
            world_road_graph, agent_position, 3000
        )
        map_ids.append(nearby_roadgraph_points.id[0].tolist())
    
    # Sort map IDs
    sorted_map_ids = []
    if map_ids and len(map_ids[0]) > 0:
        for i in range(len(map_ids[0])):
            for j in range(len(map_ids)):
                if i < len(map_ids[j]) and map_ids[j][i] > 0 and map_ids[j][i] not in sorted_map_ids:
                    sorted_map_ids.append(map_ids[j][i])
    
    # Extract roadgraph properties
    roadgraph_points_x = np.asarray(global_road_graph.x[world_idx])
    roadgraph_points_y = np.asarray(global_road_graph.y[world_idx])
    roadgraph_points_heading = np.asarray(global_road_graph.orientation[world_idx])
    roadgraph_points_types = np.asarray(global_road_graph.vbd_type[world_idx])
    road_graph_points_ids = np.asarray(global_road_graph.id[world_idx])
    
    # Build polylines
    polylines = []
    for id in sorted_map_ids:
        id_mask = road_graph_points_ids == id
        p_x = roadgraph_points_x[id_mask]
        p_y = roadgraph_points_y[id_mask]
        heading = roadgraph_points_heading[id_mask]
        lane_type = roadgraph_points_types[id_mask]
        traffic_light_state = np.zeros_like(lane_type)
        
        polyline = np.stack([p_x, p_y, heading, traffic_light_state, lane_type], axis=1)
        polyline_len = polyline.shape[0]
        
        # Sample points evenly
        sampled_points = np.linspace(0, polyline_len - 1, num_points_polyline, dtype=np.int32)
        cur_polyline = np.take(polyline, sampled_points, axis=0)
        polylines.append(cur_polyline)
    
    # Post-processing polylines
    if len(polylines) > 0:
        polylines = np.stack(polylines, axis=0)
        polylines_valid = np.ones((polylines.shape[0],), dtype=np.int32)
    else:
        polylines = np.zeros((1, num_points_polyline, 5), dtype=np.float32)
        polylines_valid = np.zeros((1,), dtype=np.int32)
    
    # Ensure polylines fit max_polylines limit
    if polylines.shape[0] >= max_polylines:
        polylines = polylines[:max_polylines]
        polylines_valid = polylines_valid[:max_polylines]
    else:
        polylines = np.pad(
            polylines,
            ((0, max_polylines - polylines.shape[0]), (0, 0), (0, 0))
        )
        polylines_valid = np.pad(
            polylines_valid, (0, max_polylines - polylines_valid.shape[0])
        )
    
    return polylines, polylines_valid

def extract_world_data(data, world_idx):
    """
    Extract data for a specific world from batched inputs.
    """
    # For simple tensor attributes
    if isinstance(data, torch.Tensor):
        return data[world_idx:world_idx+1]  # Keep dimension for compatibility
    
    # For custom objects with tensor attributes
    if hasattr(data, 'x') and hasattr(data, 'y'):  # GlobalRoadGraphPoints-like object
        world_data = type(data).__new__(type(data))
        # Copy tensor attributes with slicing for world_idx
        for attr_name in dir(data):
            if attr_name.startswith('__'):
                continue
            attr = getattr(data, attr_name)
            if isinstance(attr, torch.Tensor) and attr.dim() > 0:
                setattr(world_data, attr_name, attr[world_idx:world_idx+1])
            else:
                setattr(world_data, attr_name, attr)
        return world_data
    
    # For other types, return as is
    return data

def process_scenario_data(
    max_controlled_agents,
    controlled_agent_mask,
    global_agent_obs,
    global_road_graph,
    log_trajectory,
    init_steps,
    episode_len,
    raw_agent_types,
    metadata,
    max_polylines=256,
    num_points_polyline=30,
):
    """
    Process scenario data for multiple worlds in parallel where possible.
    First dim of all inputs and outputs is num_worlds.
    """
    num_worlds = global_agent_obs.vehicle_length.shape[0]
    
    # Process all agents across all worlds in a vectorized way
    agents_history, agents_future, agents_interested, agents_type, agents_id = process_agents_vectorized(
        num_worlds, max_controlled_agents, init_steps, global_agent_obs, 
        log_trajectory, metadata, raw_agent_types, controlled_agent_mask
    )
    
    # Initialize output tensors with batch dimension
    all_polylines = np.zeros((num_worlds, max_polylines, num_points_polyline, 5), dtype=np.float32)
    all_polylines_valid = np.zeros((num_worlds, max_polylines), dtype=np.int32)
    all_traffic_light_points = np.zeros((num_worlds, 16, 3), dtype=np.float32)
    all_relations = np.zeros((num_worlds, agents_history.shape[1] + max_polylines + 16, 
                             agents_history.shape[1] + max_polylines + 16, 3), dtype=np.float32)
    
    # Process roadgraph data for each world (parallel processing isn't efficient here due to variable data dependencies)
    for w in range(num_worlds):
        world_polylines, world_polylines_valid = process_world_roadgraph(
            global_road_graph, w, agents_history[w], agents_interested[w],
            max_polylines, num_points_polyline
        )
        
        all_polylines[w] = world_polylines
        all_polylines_valid[w] = world_polylines_valid
        
        # Calculate relations for this world
        all_relations[w] = calculate_relations(
            agents_history[w],
            all_polylines[w],
            all_traffic_light_points[w]
        )
    
    # Prepare the output dictionary with batch dimensions
    data_dict = {
        "agents_history": np.float32(agents_history),
        "agents_interested": np.int32(agents_interested),
        "agents_type": np.int32(agents_type),
        "agents_future": np.float32(agents_future),
        "traffic_light_points": np.float32(all_traffic_light_points),
        "polylines": np.float32(all_polylines),
        "polylines_valid": np.int32(all_polylines_valid),
        "relations": np.float32(all_relations),
        "agents_id": np.int32(agents_id),
    }
    
    # Convert to PyTorch tensors
    torch_dict = {
        key: torch.from_numpy(value) 
        for key, value in data_dict.items()
    }
    torch_dict["anchors"] = torch.zeros(num_worlds, 32, 64, 2)  # Batch-sized placeholder
    
    return torch_dict

def sample_to_action():
    """Todo: Implement this function."""
    pass
