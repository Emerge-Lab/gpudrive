"""Functions used for processing roadgraph data and other features for VBD."""
import torch
import numpy as np
import gpudrive
from integrations.models.vbd.data.data_utils import calculate_relations
from pygpudrive.datatypes.roadgraph import GlobalRoadGraphPoints

def process_scenario_data(
    max_controlled_agents,
    controlled_agent_mask,
    global_agent_obs,
    global_road_graph,
    log_trajectory,
    init_steps,
    episode_len,
    raw_agent_types,
    max_polylines=256,
    num_points_polyline=30,
):
    """Process the scenario data for Versatile Behavior Diffusion."""

    # History of agents (init_steps + 1)
    agents_history = construct_agent_history(
        init_steps=init_steps,
        controlled_agent_mask=controlled_agent_mask,
        max_cont_agents=max_controlled_agents,
        global_agent_obs=global_agent_obs,
        log_trajectory=log_trajectory,
    )

    # 10 if we are controlling the agent, 1 otherwise
    agents_interested = torch.ones(max_controlled_agents)
    agents_interested[controlled_agent_mask] = 10

    # Type of agents: 0 for None, 1 for Vehicle, 2 for Pedestrian, 3 for Cyclist
    agents_type = torch.zeros(max_controlled_agents).long()
    agents_type[
        raw_agent_types == int(gpudrive.EntityType.Vehicle)
    ] = 1  # Vehicle
    agents_type[
        raw_agent_types == int(gpudrive.EntityType.Pedestrian)
    ] = 2  # Pedestrian
    agents_type[
        raw_agent_types == int(gpudrive.EntityType.Cyclist)
    ] = 3  # Cyclist

    # Now create the agents future logs
    agents_future = torch.cat(
        [
            log_trajectory.pos_xy[0, :, init_steps:, :],  
            log_trajectory.yaw[0, :, init_steps:, :],
            log_trajectory.vel_xy[0, :, init_steps:, :], 
        ],
        dim=-1,
    )

    # Set all invalid agent values to zero
    agents_future[~controlled_agent_mask, :, :] = 0

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
    sorted_map_ids = []
    for i in range(nearby_roadgraph_points.num_points):
        for j in range(len(map_ids)):
            if map_ids[j][i] > 0 and map_ids[j][i] not in sorted_map_ids:
                sorted_map_ids.append(map_ids[j][i])
    polylines = []
    roadgraph_points_x = np.asarray(global_road_graph.x[0])
    roadgraph_points_y = np.asarray(global_road_graph.y[0])
    roadgraph_points_heading = np.asarray(global_road_graph.orientation[0])
    roadgraph_points_types = np.asarray(global_road_graph.type[0])
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
    traffic_light_points = torch.zeros((16, 3))

    # Controlled agents
    agents_id = torch.nonzero(controlled_agent_mask).permute(1, 0)

    # Compute relations at the end
    relations = calculate_relations(
        agents_history,
        polylines,
        traffic_light_points,
    )

    data_dict = {
        "agents_history": agents_history,
        "agents_interested": agents_interested,
        "agents_type": agents_type.long(),
        "agents_future": agents_future,
        "traffic_light_points": traffic_light_points,
        "polylines": polylines,
        "polylines_valid": polylines_valid,
        "relations": torch.Tensor(relations),
        "agents_id": agents_id,
        "anchors": torch.zeros((1, 32, 64, 2)),  # Placeholder, not used
    }

    return data_dict


def construct_agent_history(
    init_steps,
    controlled_agent_mask,
    max_cont_agents,
    global_agent_obs,
    log_trajectory,
):
    """Get the agent trajectory feature information."""

    agents_history = torch.cat(
        [
            log_trajectory.pos_xy[0, :, : init_steps + 1, :],  
            log_trajectory.yaw[0, :, : init_steps + 1, :],
            log_trajectory.vel_xy[0, :, : init_steps + 1, :], 
            global_agent_obs.vehicle_length[0].unsqueeze(-1).expand(-1, init_steps + 1).unsqueeze(-1),
            global_agent_obs.vehicle_width[0].unsqueeze(-1).expand(-1, init_steps + 1).unsqueeze(-1),
            torch.ones((max_cont_agents, init_steps + 1)).unsqueeze(-1),
        ],
        dim=-1,
    )
    
    # Zero out the agents that are not controlled
    agents_history[~controlled_agent_mask, :, :] = 0.0

    return agents_history

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
        filtered_type = global_road_graph.type[0][top_idx]
        filtered_id = global_road_graph.id[0][top_idx]

        # Stack the filtered attributes to form a new roadgraph tensor
        filtered_tensor = torch.stack([
            filtered_xy[..., 0],
            filtered_xy[..., 1],
            filtered_length,
            filtered_width,
            filtered_height,
            filtered_orientation,
            torch.zeros_like(filtered_length),
            filtered_id,
            filtered_type
        ], dim=-1)

        return GlobalRoadGraphPoints(filtered_tensor.clone())
    else:
        return global_road_graph


def sample_to_action():
    """Todo: Implement this function."""
    pass
