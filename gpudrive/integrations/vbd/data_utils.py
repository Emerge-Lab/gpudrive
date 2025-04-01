"""Functions used for processing roadgraph data and other features for VBD."""
import torch
import madrona_gpudrive
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints


def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi].
    Args:
        angle (torch.Tensor): The input angle.
    Returns:
        torch.Tensor: The wrapped angle.
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


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
        device = global_road_graph.xy.device
        roadgraph_xy = global_road_graph.xy[0]  # Shape [N, 2]
        
        # Use torch functions for distance calculation
        # reference_points shape is [..., 2]
        expanded_ref = reference_points.unsqueeze(-2)  # Shape [..., 1, 2]
        # roadgraph_xy shape is [N, 2]
        expanded_roadgraph = roadgraph_xy.unsqueeze(0)  # Shape [1, N, 2]
        # Calculate distance using broadcasting
        distances = torch.norm(expanded_ref - expanded_roadgraph, dim=-1)  # Shape [..., N]
        
        # Create mask for valid points (id > 0)
        valid_mask = global_road_graph.id[0] > 0
        valid_distances = torch.where(valid_mask, distances, torch.tensor(float('inf'), device=device))
        
        # Find topk indices - we want the smallest distances
        # top_idx will have shape [..., topk]
        _, top_idx = torch.topk(valid_distances, k=topk, largest=False, dim=-1)
        
        # Handle the case where reference_points might have leading batch dimensions
        batch_dims = reference_points.size()[:-1]  # Get all dimensions except the last one
        
        # For each attribute, we'll gather the topk values using advanced indexing
        
        # For xy coordinates, we need to handle the last dimension (size 2) separately
        batch_size = torch.prod(torch.tensor(batch_dims), dtype=torch.int64) if batch_dims else 1
        
        # Reshape for easier gathering
        flat_top_idx = top_idx.view(batch_size, topk)  # Shape [batch_size, topk]
        
        # Create output tensors
        filtered_xy = torch.zeros((batch_size, topk, 2), device=device)
        filtered_length = torch.zeros((batch_size, topk), device=device)
        filtered_width = torch.zeros((batch_size, topk), device=device)
        filtered_height = torch.zeros((batch_size, topk), device=device)
        filtered_orientation = torch.zeros((batch_size, topk), device=device)
        filtered_type = torch.zeros((batch_size, topk), device=device)
        filtered_id = torch.zeros((batch_size, topk), device=device)
        
        # Gather each value efficiently
        for b in range(batch_size):
            indices = flat_top_idx[b]  # Shape [topk]
            filtered_xy[b] = roadgraph_xy[indices]  # Shape [topk, 2]
            filtered_length[b] = global_road_graph.segment_length[0][indices]
            filtered_width[b] = global_road_graph.segment_width[0][indices]
            filtered_height[b] = global_road_graph.segment_height[0][indices]
            filtered_orientation[b] = global_road_graph.orientation[0][indices]
            filtered_type[b] = global_road_graph.vbd_type[0][indices]
            filtered_id[b] = global_road_graph.id[0][indices]
        
        # Reshape back to original batch dimensions
        if batch_dims:
            new_shape = batch_dims + (topk, 2)
            filtered_xy = filtered_xy.view(new_shape)
            
            new_shape = batch_dims + (topk,)
            filtered_length = filtered_length.view(new_shape)
            filtered_width = filtered_width.view(new_shape)
            filtered_height = filtered_height.view(new_shape)
            filtered_orientation = filtered_orientation.view(new_shape)
            filtered_type = filtered_type.view(new_shape)
            filtered_id = filtered_id.view(new_shape)
        
        # Stack the filtered attributes to form a new roadgraph tensor
        # For simplicity, we'll handle the case where batch_size is 1 (most common case)
        if batch_size == 1 and not batch_dims:
            filtered_tensor = torch.stack(
                [
                    filtered_xy[0, :, 0],
                    filtered_xy[0, :, 1],
                    filtered_length[0],
                    filtered_width[0],
                    filtered_height[0],
                    filtered_orientation[0],
                    torch.zeros_like(filtered_length[0]),
                    filtered_id[0],
                    filtered_type[0]
                ],
                dim=-1
            ).unsqueeze(0)  # Add batch dimension back
            
        else:
            # More complex case with multiple batch dimensions
            # This would need careful reshaping to handle correctly
            filtered_tensor = torch.zeros(batch_dims + (topk, 9), device=device)
            
            # Handle the general case with proper reshaping
            filtered_tensor[..., 0] = filtered_xy[..., 0]
            filtered_tensor[..., 1] = filtered_xy[..., 1]
            filtered_tensor[..., 2] = filtered_length
            filtered_tensor[..., 3] = filtered_width
            filtered_tensor[..., 4] = filtered_height
            filtered_tensor[..., 5] = filtered_orientation
            filtered_tensor[..., 6] = torch.zeros_like(filtered_length)
            filtered_tensor[..., 7] = filtered_id
            filtered_tensor[..., 8] = filtered_type
            
            # Add the world batch dimension if not present
            if len(filtered_tensor.shape) == 2:  # [topk, 9]
                filtered_tensor = filtered_tensor.unsqueeze(0)  # [1, topk, 9]
            
        return GlobalRoadGraphPoints(filtered_tensor.clone())
    else:
        return global_road_graph


def calculate_relations(agents, polylines, traffic_lights):
    """
    Calculate the relations between agents, polylines, and traffic lights.
    Args:
        agents (torch.Tensor): Tensor of agent positions and orientations.
        polylines (torch.Tensor): Tensor of polyline positions.
        traffic_lights (torch.Tensor): Tensor of traffic light positions.
    Returns:
        torch.Tensor: Tensor of relations between the elements.
    """
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights
    
    # Ensure all inputs are torch tensors
    device = agents.device
    
    # Prepare a single tensor to hold all elements
    all_elements = torch.cat(
        [
            agents[:, -1, :3],
            polylines[:, 0, :3],
            torch.cat(
                [traffic_lights[:, :2], torch.zeros((n_traffic_lights, 1), device=device)],
                dim=1,
            ),
        ],
        dim=0,
    )
    
    # Compute pairwise differences using broadcasting
    # Create expanded views for broadcasting
    pos1 = all_elements[:, :2].unsqueeze(1)  # Shape: [n, 1, 2]
    pos2 = all_elements[:, :2].unsqueeze(0)  # Shape: [1, n, 2]
    pos_diff = pos1 - pos2  # Broadcasting gives shape: [n, n, 2]
    
    # Compute local positions and angle differences
    cos_theta = torch.cos(all_elements[:, 2]).unsqueeze(1)
    sin_theta = torch.sin(all_elements[:, 2]).unsqueeze(1)
    
    # Calculate local coordinates
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    
    # Calculate angle differences
    theta1 = all_elements[:, 2].unsqueeze(1)  # Shape: [n, 1]
    theta2 = all_elements[:, 2].unsqueeze(0)  # Shape: [1, n]
    theta_diff = wrap_to_pi(theta1 - theta2)  # Shape: [n, n]
    
    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    traffic_mask = (torch.arange(n, device=device) >= start_idx).unsqueeze(1) | \
                   (torch.arange(n, device=device) >= start_idx).unsqueeze(0)
    theta_diff = torch.where(traffic_mask, torch.tensor(0.0, device=device), theta_diff)
    
    # Set the diagonal of the differences to a very small value
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)
    epsilon = 0.01
    local_pos_x = torch.where(diag_mask, torch.tensor(epsilon, device=device), local_pos_x)
    local_pos_y = torch.where(diag_mask, torch.tensor(epsilon, device=device), local_pos_y)
    theta_diff = torch.where(diag_mask, torch.tensor(epsilon, device=device), theta_diff)
    
    # Conditions for zero coordinates
    zero_mask = (all_elements[:, 0].unsqueeze(1) == 0) | (all_elements[:, 0].unsqueeze(0) == 0)
    
    # Initialize relations tensor
    relations = torch.stack([local_pos_x, local_pos_y, theta_diff], dim=-1)
    
    # Apply zero mask
    relations = torch.where(zero_mask.unsqueeze(-1), torch.tensor(0.0, device=device), relations)
    
    return relations


def process_agents_vectorized(num_worlds, max_cont_agents, init_steps, global_agent_obs, 
                             log_trajectory, metadata, raw_agent_types, controlled_agent_mask):
    """
    Vectorized function to process agent data across multiple worlds.
    Using controlled_agent_mask instead of SDC proximity.
    All tensor operations are performed in PyTorch.
    """
    device = controlled_agent_mask.device
    
    # Initialize output tensors with batch dimension
    agents_history = torch.zeros((num_worlds, max_cont_agents, init_steps + 1, 8), 
                                dtype=torch.float32, device=device)
    agents_type = torch.zeros((num_worlds, max_cont_agents), 
                             dtype=torch.int32, device=device)
    agents_interested = torch.zeros((num_worlds, max_cont_agents), 
                                   dtype=torch.int32, device=device)
    agents_future = torch.zeros(
        (num_worlds, max_cont_agents, log_trajectory.pos_xy.shape[2] - init_steps, 5),
        dtype=torch.float32, device=device
    )
    agents_id = torch.zeros((num_worlds, max_cont_agents), 
                           dtype=torch.int32, device=device)
    
    # Process each world using controlled_agent_mask
    for w in range(num_worlds):
        # Get indices of controlled agents using torch.where
        controlled_indices = torch.where(controlled_agent_mask[w])[0]
        
        # Sort by agent ID for consistency
        sorted_agent_indices, _ = torch.sort(controlled_indices)
        
        # Handle case where we have fewer controlled agents than max_cont_agents
        if len(sorted_agent_indices) < max_cont_agents:
            # Pad with -1 i.e. invalid agent index
            padded_indices = torch.full((max_cont_agents,), -1, 
                                      dtype=torch.int32, device=device)
            padded_indices[:len(sorted_agent_indices)] = sorted_agent_indices
            sorted_agent_indices = padded_indices
        
        # Store agent indices
        agents_id[w] = sorted_agent_indices
        
        # Process each agent for this world
        for i, a in enumerate(sorted_agent_indices):
            if a == -1:
                break
                
            # Get agent type
            if isinstance(raw_agent_types, list):
                agent_type = raw_agent_types[w][a]
            else:  # Assume it's a tensor
                agent_type = raw_agent_types[w, a]
                
            valid = log_trajectory.valids[w, a, init_steps]
            
            if valid.item() != 1:
                agents_interested[w, i] = 0
                continue
                
            if metadata.tracks_to_predict[w, a] or metadata.objects_of_interest[w, a]:
                agents_interested[w, i] = 10
            else:
                agents_interested[w, i] = 1
            
            agents_type[w, i] = agent_type
            
            # Stack history data using efficient torch operations
            # Create a tensor with the right shape directly
            pos_x = log_trajectory.pos_xy[w, a, :init_steps+1, 0]
            pos_y = log_trajectory.pos_xy[w, a, :init_steps+1, 1]
            yaw = log_trajectory.yaw[w, a, :init_steps+1, 0]
            vel_x = log_trajectory.vel_xy[w, a, :init_steps+1, 0]
            vel_y = log_trajectory.vel_xy[w, a, :init_steps+1, 1]
            veh_length = global_agent_obs.vehicle_length[w, a].repeat(init_steps + 1)
            veh_width = global_agent_obs.vehicle_width[w, a].repeat(init_steps + 1)
            veh_height = global_agent_obs.vehicle_height[w, a].repeat(init_steps + 1)
            
            # Stack along the feature dimension (last dimension)
            history_data = torch.stack([pos_x, pos_y, yaw, vel_x, vel_y, 
                                       veh_length, veh_width, veh_height], dim=-1)
            
            agents_history[w, i] = history_data
            
            # Apply mask correctly by matching dimensions
            mask = log_trajectory.valids[w, a, :init_steps+1]
            # Reshape mask to [init_steps+1, 1] for proper broadcasting across features
            mask = mask.view(-1, 1)
            agents_history[w, i] = agents_history[w, i] * mask
            
            # Use the same approach for future data - efficient vectorized operations
            pos_x_fut = log_trajectory.pos_xy[w, a, init_steps:, 0]
            pos_y_fut = log_trajectory.pos_xy[w, a, init_steps:, 1]
            yaw_fut = log_trajectory.yaw[w, a, init_steps:, 0]
            vel_x_fut = log_trajectory.vel_xy[w, a, init_steps:, 0]
            vel_y_fut = log_trajectory.vel_xy[w, a, init_steps:, 1]
            
            # Stack along the feature dimension (last dimension)
            future_data = torch.stack([pos_x_fut, pos_y_fut, yaw_fut, vel_x_fut, vel_y_fut], dim=-1)
            
            agents_future[w, i] = future_data
            
            # Apply mask for future data correctly
            mask_future = log_trajectory.valids[w, a, init_steps:]
            # Reshape mask to [future_len, 1] for proper broadcasting across features
            mask_future = mask_future.view(-1, 1)
            agents_future[w, i] = agents_future[w, i] * mask_future
    
    # Map agent types for all worlds at once using torch operations
    mapped_agents_type = torch.zeros_like(agents_type)
    mapped_agents_type[agents_type == int(madrona_gpudrive.EntityType.Vehicle)] = 1
    mapped_agents_type[agents_type == int(madrona_gpudrive.EntityType.Pedestrian)] = 2
    mapped_agents_type[agents_type == int(madrona_gpudrive.EntityType.Cyclist)] = 3
    
    return agents_history, agents_future, agents_interested, mapped_agents_type, agents_id


def process_world_roadgraph(global_road_graph, world_idx, agents_history, agents_interested, 
                           max_polylines, num_points_polyline):
    """
    Process the roadgraph for a single world.
    All operations use PyTorch tensors.
    """
    device = global_road_graph.x.device
    
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
    
    # Extract roadgraph properties as tensors
    roadgraph_points_x = global_road_graph.x[world_idx]
    roadgraph_points_y = global_road_graph.y[world_idx]
    roadgraph_points_heading = global_road_graph.orientation[world_idx]
    roadgraph_points_types = global_road_graph.vbd_type[world_idx]
    road_graph_points_ids = global_road_graph.id[world_idx]
    
    # Build polylines using PyTorch operations
    polylines = []
    for id in sorted_map_ids:
        id_mask = road_graph_points_ids == id
        p_x = roadgraph_points_x[id_mask]
        p_y = roadgraph_points_y[id_mask]
        heading = roadgraph_points_heading[id_mask]
        lane_type = roadgraph_points_types[id_mask]
        traffic_light_state = torch.zeros_like(lane_type)
        
        polyline = torch.stack([p_x, p_y, heading, traffic_light_state, lane_type], dim=1)
        polyline_len = polyline.shape[0]
        
        # Sample points evenly
        if polyline_len > 0:
            # Create indices for sampling
            indices = torch.linspace(0, polyline_len - 1, num_points_polyline, device=device).long()
            cur_polyline = polyline[indices]
            polylines.append(cur_polyline)
    
    # Post-processing polylines
    if len(polylines) > 0:
        polylines = torch.stack(polylines, dim=0)
        polylines_valid = torch.ones((polylines.shape[0],), dtype=torch.int32, device=device)
    else:
        polylines = torch.zeros((1, num_points_polyline, 5), dtype=torch.float32, device=device)
        polylines_valid = torch.zeros((1,), dtype=torch.int32, device=device)
    
    # Ensure polylines fit max_polylines limit
    if polylines.shape[0] >= max_polylines:
        polylines = polylines[:max_polylines]
        polylines_valid = polylines_valid[:max_polylines]
    else:
        # Create zero padding tensor
        padding = torch.zeros(
            (max_polylines - polylines.shape[0], num_points_polyline, 5),
            dtype=torch.float32, device=device
        )
        polylines = torch.cat([polylines, padding], dim=0)
        
        # Pad valid flags
        valid_padding = torch.zeros(max_polylines - polylines_valid.shape[0], 
                                  dtype=torch.int32, device=device)
        polylines_valid = torch.cat([polylines_valid, valid_padding], dim=0)
    
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
    All operations use PyTorch tensors.
    """
    num_worlds = global_agent_obs.vehicle_length.shape[0]
    device = global_agent_obs.vehicle_length.device
    
    # Process all agents across all worlds in a vectorized way
    agents_history, agents_future, agents_interested, agents_type, agents_id = process_agents_vectorized(
        num_worlds, max_controlled_agents, init_steps, global_agent_obs,
        log_trajectory, metadata, raw_agent_types, controlled_agent_mask
    )
    
    # Initialize output tensors with batch dimension
    all_polylines = torch.zeros((num_worlds, max_polylines, num_points_polyline, 5), 
                               dtype=torch.float32, device=device)
    all_polylines_valid = torch.zeros((num_worlds, max_polylines), 
                                     dtype=torch.int32, device=device)
    all_traffic_light_points = torch.zeros((num_worlds, 16, 3), 
                                          dtype=torch.float32, device=device)
    all_relations = torch.zeros(
        (num_worlds, agents_history.shape[1] + max_polylines + 16,
         agents_history.shape[1] + max_polylines + 16, 3), 
        dtype=torch.float32, device=device
    )
    
    # Process roadgraph data for each world
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
    
    # Prepare the output dictionary
    data_dict = {
        "agents_history": agents_history,
        "agents_interested": agents_interested,
        "agents_type": agents_type,
        "agents_future": agents_future,
        "traffic_light_points": all_traffic_light_points,
        "polylines": all_polylines,
        "polylines_valid": all_polylines_valid,
        "relations": all_relations,
        "agents_id": agents_id,
        "anchors": torch.zeros(num_worlds, 32, 64, 2, device=device)  # Batch-sized placeholder
    }
    
    return data_dict


def sample_to_action():
    """Todo: Implement this function."""
    pass