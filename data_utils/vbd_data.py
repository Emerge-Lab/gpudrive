"""Functions used for processing roadgraph data and other features for VBD."""
import torch
import numpy as np
import gpudrive
from pygpudrive.datatypes.roadgraph import GlobalRoadGraphPoints

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


def data_process_agent(
    init_steps,
    max_cont_agents,
    global_agent_obs,
    log_trajectory,
    metadata,
    agent_types,
):
    """
    Process the data for surrounding agents in a given scenario.

    Args:
        
    Returns:
        tuple: A tuple containing the processed agent data, including:
            - agents_history (ndarray): The history of agent trajectories. Shape: (max_object, history_length, 8)
            - agents_future (ndarray): The future agent trajectories. Shape: (max_object, future_length, 5)
            - agents_interested (ndarray): The interest level of agents. Shape: (max_object,)
            - agents_type (ndarray): The type of agents. Shape: (max_object,)
    """

    sdc_index = np.where(metadata.isSdc[0] == 1)[0][0]
    sdc_position = np.asarray(log_trajectory.pos_xy[0, sdc_index, init_steps, :])
    agent_positions = np.asarray(log_trajectory.pos_xy[0, :, init_steps])
    distance_to_sdc = np.linalg.norm( agent_positions - sdc_position, axis=-1)
    agent_indices = np.argsort(distance_to_sdc)[:max_cont_agents]
    sorted_agent_indices = np.sort(agent_indices)

    agents_history = np.zeros(
        (max_cont_agents, init_steps + 1, 8), dtype=np.float32
    )
    agents_type = np.zeros((max_cont_agents,), dtype=np.int32)
    agents_interested = np.zeros((max_cont_agents,), dtype=np.int32)
    agents_future = np.zeros(
        (max_cont_agents, log_trajectory.pos_xy.shape[-2] - init_steps, 5),
        dtype=np.float32,
    )

    for i, a in enumerate(sorted_agent_indices):
        agent_type = agent_types[a]
        valid = log_trajectory.valids[0, a, init_steps]

        if valid.item() != 1:
            agents_interested[i] = 0
            continue

        if metadata.isModeled[0, a] or metadata.isOfInterest[0, a]:
            agents_interested[i] = 10
        else:
            agents_interested[i] = 1
        
        agents_type[i] = agent_type
        agents_history[i] = torch.column_stack(
            [
                log_trajectory.pos_xy[0, a, :init_steps+1, 0],
                log_trajectory.pos_xy[0, a, :init_steps+1, 1],
                log_trajectory.yaw[0, a, :init_steps+1, 0],
                log_trajectory.vel_xy[0, a, :init_steps+1, 0],
                log_trajectory.vel_xy[0, a, :init_steps+1, 1],
                global_agent_obs.vehicle_length[0, a].repeat(init_steps + 1),
                global_agent_obs.vehicle_width[0, a].repeat(init_steps + 1),
                global_agent_obs.vehicle_height[0, a].repeat(init_steps + 1),
            ],
        ).numpy()

        mask = log_trajectory.valids[0, a, :init_steps+1].numpy()
        agents_history[i] *= mask

        agents_future[i] = torch.column_stack(
            [
                log_trajectory.pos_xy[0, a, init_steps:, 0],
                log_trajectory.pos_xy[0, a, init_steps:, 1],
                log_trajectory.yaw[0, a, init_steps:, 0],
                log_trajectory.vel_xy[0, a, init_steps:, 0],
                log_trajectory.vel_xy[0, a, init_steps:, 1],
            ],
        ).numpy()

        mask = log_trajectory.valids[0, a, init_steps:].numpy()
        agents_future[i] *= mask

    # Type of agents: 0 for None, 1 for Vehicle, 2 for Pedestrian, 3 for Cyclist
    mapped_agents_type = np.zeros_like(agents_type)
    mapped_agents_type[agents_type == int(gpudrive.EntityType.Vehicle)] = 1    # Vehicle
    mapped_agents_type[agents_type == int(gpudrive.EntityType.Pedestrian)] = 2 # Pedestrian
    mapped_agents_type[agents_type == int(gpudrive.EntityType.Cyclist)] = 3    # Cyclist


    # Zero out the agents that are not controlled
    # agents_history[~controlled_agent_mask, :, :] = 0.0

    return (
        agents_history,
        agents_future,
        agents_interested,
        mapped_agents_type,
        sorted_agent_indices,
    )

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
    """Process the scenario data for Versatile Behavior Diffusion."""

    (
        agents_history,
        agents_future,
        agents_interested,
        agents_type,
        agents_id,
    ) = data_process_agent(
        init_steps=init_steps,
        max_cont_agents=max_controlled_agents,
        global_agent_obs=global_agent_obs,
        log_trajectory=log_trajectory,
        metadata=metadata,
        agent_types=raw_agent_types
    )

    # Set all invalid agent values to zero
    # agents_future[~controlled_agent_mask, :, :] = 0

    # Global polylines tensor: Shape (256, 30, 5)
    map_ids = []
    current_valid = agents_interested > 0

    for agent in range(agents_history.shape[0]):
        if not current_valid[agent]:
            continue
        agent_position = agents_history[agent, -1, :2]
        nearby_roadgraph_points = filter_topk_roadgraph_points(
            global_road_graph, agent_position, 3000
        )
        map_ids.append(nearby_roadgraph_points.id[0].tolist())

    # sort map ids
    sorted_map_ids = []
    for i in range(nearby_roadgraph_points.num_points):
        for j in range(len(map_ids)):
            if map_ids[j][i] > 0 and map_ids[j][i] not in sorted_map_ids:
                sorted_map_ids.append(map_ids[j][i])

    # get shared map polylines
    # polyline feature: x, y, heading, traffic_light, type
    polylines = []
    roadgraph_points_x = np.asarray(global_road_graph.x[0])
    roadgraph_points_y = np.asarray(global_road_graph.y[0])
    roadgraph_points_heading = np.asarray(global_road_graph.orientation[0])
    roadgraph_points_types = np.asarray(global_road_graph.vbd_type[0])
    road_graph_points_ids = np.asarray(global_road_graph.id[0])
    
    for id in sorted_map_ids:
        p_x = roadgraph_points_x[road_graph_points_ids == id]
        p_y = roadgraph_points_y[road_graph_points_ids == id]
        heading = roadgraph_points_heading[road_graph_points_ids == id]
        lane_type = roadgraph_points_types[road_graph_points_ids == id]
        traffic_light_state = np.zeros_like(lane_type)
        polyline = np.stack(
            [p_x, p_y, heading, traffic_light_state, lane_type], axis=1
        )
        # sample points and fill into fixed-size array
        polyline_len = polyline.shape[0]
        sampled_points = np.linspace(
            0, polyline_len  - 1, num_points_polyline, dtype=np.int32
        )
        cur_polyline = np.take(polyline, sampled_points, axis=0)
        polylines.append(cur_polyline)

    #post processing polylines
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
        polylines = np.pad(
            polylines,
            ((0, max_polylines - polylines.shape[0]), (0, 0), (0, 0))
        )
        polylines_valid = np.pad(
            polylines_valid, (0, max_polylines - polylines_valid.shape[0])
        )

    # Empty (16, 3)
    traffic_light_points = np.zeros((16, 3))

    # Compute relations at the end
    relations = calculate_relations(
        agents_history,
        polylines,
        traffic_light_points,
    )
    relations = np.asarray(relations)

    data_dict = {
        "agents_history": np.float32(agents_history),
        "agents_interested": np.int32(agents_interested),
        "agents_type": np.int32(agents_type),
        "agents_future": np.float32(agents_future),
        "traffic_light_points": np.float32(traffic_light_points),
        "polylines": np.float32(polylines),
        "polylines_valid": np.int32(polylines_valid),
        "relations": np.float32(relations),
        "agents_id": np.int32(agents_id),
        "anchors": np.zeros((1, 32, 64, 2)),  # Placeholder, not used
    }

    return data_dict


def sample_to_action():
    """Todo: Implement this function."""
    pass
