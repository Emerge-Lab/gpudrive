"""Functions used for processing roadgraph data and other features for VBD."""
import torch
import numpy as np
import gpudrive
from integrations.models.vbd.data.data_utils import calculate_relations


def restore_mean(x, y, mean_x, mean_y):
    """
    In GPUDrive, everything is centered at zero by subtracting the mean.
    This function reapplies the mean to go back to the original coordinates.
    The mean (xyz) is exported per world as world_means_tensor.

    Args:
        x (torch.Tensor): x coordinates
        y (torch.Tensor): y coordinates
        mean_x (torch.Tensor): mean of x coordinates. Shape: (num_worlds, 1)
        mean_y (torch.Tensor): mean of y coordinates. Shape: (num_worlds, 1)
    """
    return x + mean_x, y + mean_y


def process_scenario_data(
    num_envs,
    max_controlled_agents,
    controlled_agent_mask,
    global_agent_obs,
    global_road_graph,
    local_road_graph,
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
        num_envs=num_envs,
        global_agent_obs=global_agent_obs,
        log_trajectory=log_trajectory,
    )

    # Now create the agents future logs
    agents_future = torch.zeros(
        (num_envs, max_controlled_agents, episode_len - init_steps, 5)
    )
    agents_future[:, :, :, 0:2] = log_trajectory.pos_xy[:, :, init_steps:, :]
    agents_future[:, :, :, 2] = log_trajectory.yaws[:, :, init_steps:, :].squeeze(-1)
    agents_future[:, :, :, 3:5] = log_trajectory.vel_xy[:, :, init_steps:, :]

    # Set all invalid agent values to zero
    agents_future[~controlled_agent_mask, :, :] = 0

    # Type of agents: 0 for None, 1 for Vehicle, 2 for Pedestrian, 3 for Cyclist
    agents_type = torch.zeros([num_envs, max_controlled_agents]).long()
    agents_type[
        raw_agent_types == int(gpudrive.EntityType.Vehicle)
    ] = 1  # Vehicle
    agents_type[
        raw_agent_types == int(gpudrive.EntityType.Pedestrian)
    ] = 2  # Pedestrian
    agents_type[
        raw_agent_types == int(gpudrive.EntityType.Cyclist)
    ] = 3  # Cyclist

    # 10 if we are controlling the agent, 1 otherwise
    agents_interested = torch.ones([num_envs, max_controlled_agents])
    agents_interested[controlled_agent_mask] = 10

    # Global polylines tensor: Shape (256, 30, 5)
    polylines, polylines_valid = construct_polylines(
        global_road_graph,
        local_road_graph,
        num_envs,
        controlled_agent_mask,
        max_polylines,
        num_points_polyline,
    )

    # Empty (16, 3)
    traffic_light_points = torch.zeros((num_envs, 16, 3))

    # Controlled agents
    agents_id = torch.nonzero(controlled_agent_mask[0, :]).permute(1, 0)

    # Compute relations at the end
    relations = calculate_relations(
        agents_history.squeeze(0),
        polylines.squeeze(0),
        traffic_light_points.squeeze(0),
    )

    data_dict = {
        "agents_history": agents_history,
        "agents_interested": agents_interested,
        "agents_type": agents_type.long(),
        "agents_future": agents_future,
        "traffic_light_points": traffic_light_points,
        "polylines": polylines.unsqueeze(0),
        "polylines_valid": polylines_valid.unsqueeze(0),
        "relations": torch.Tensor(relations).unsqueeze(0),
        "agents_id": agents_id,
        "anchors": torch.zeros((1, 32, 64, 2)),  # Placeholder, not used
    }

    return data_dict


def construct_agent_history(
    init_steps,
    controlled_agent_mask,
    max_cont_agents,
    num_envs,
    global_agent_obs,
    log_trajectory,
):
    """Get the agent trajectory feature information."""

    agents_history = torch.cat(
        [
            log_trajectory.pos_xy[:, :, : init_steps + 1, :],  
            log_trajectory.yaw[:, :, : init_steps + 1, :],
            log_trajectory.vel_xy[:, :, : init_steps + 1, :], 
            global_agent_obs.vehicle_length.unsqueeze(-1).expand(-1, -1, init_steps + 1).unsqueeze(-1),
            global_agent_obs.vehicle_width.unsqueeze(-1).expand(-1, -1, init_steps + 1).unsqueeze(-1),
            torch.ones((num_envs, max_cont_agents, init_steps + 1)).unsqueeze(-1),
        ],
        dim=-1,
    )
    
    # Zero out the agents that are not controlled
    agents_history[~controlled_agent_mask, :, :] = 0.0

    return agents_history


def construct_polylines(
    global_road_graph,
    local_road_graph,
    num_envs,
    controlled_agent_mask,
    max_polylines,
    num_points_polyline,
):
    """Get the global polylines information."""

    # Obtain the K road graph IDs closest to the controlled agents
    nearby_road_graph_ids = []
    valid_rg_id_mask = (local_road_graph[:, :, :, 7] != -1) & (
        local_road_graph[:, :, :, 7] != 0
    )
    close_and_valid_mask = valid_rg_id_mask & controlled_agent_mask.unsqueeze(
        -1
    ).expand(-1, -1, max_polylines)

    for env_idx in range(num_envs):
        local_road_ids_env = local_road_graph[env_idx, :, :, 7].long()
        # Store the road object IDs that are valid and close to controlled agents
        valid_and_close_road_ids = local_road_ids_env[
            close_and_valid_mask[env_idx, :]
        ].unique()
        nearby_road_graph_ids.append(valid_and_close_road_ids.tolist())

    # Sort the road graph IDs
    # TODO(dc): Is this necessary? Think our road graph is already sorted
    sorted_map_ids = nearby_road_graph_ids

    # get shared map polylines
    # polyline feature: x, y, heading, traffic_light, type
    for env_idx in range(num_envs):
        polylines = []
        sorted_map_ids_env = sorted_map_ids[env_idx]

        for id in sorted_map_ids_env:

            # Select the road points that belong to this road ID
            selected_road_point_idx = torch.where(
                global_road_graph[env_idx, :, 7] == id
            )[0]

            # Get polyline
            p_x = global_road_graph[env_idx, selected_road_point_idx, 0]
            p_y = global_road_graph[env_idx, selected_road_point_idx, 1]
            heading = global_road_graph[env_idx, selected_road_point_idx, 5]
            lane_type = global_road_graph[
                env_idx, selected_road_point_idx, 8
            ].long()
            traffic_light_state = torch.zeros_like(lane_type)

            polyline = torch.stack(
                [p_x, p_y, heading, traffic_light_state, lane_type], axis=1
            )
            # Sample points and fill into fixed-size array
            polyline_len = polyline.shape[0]
            sampled_points = torch.linspace(
                0, polyline_len - 1, num_points_polyline, dtype=torch.int32
            )
            cur_polyline = torch.index_select(
                polyline, dim=0, index=sampled_points
            )
            polylines.append(cur_polyline)

        # Post processing polylines
        if len(polylines) > 0:
            polylines = torch.stack(polylines, axis=0)
            polylines_valid = torch.ones(
                (polylines.shape[0],), dtype=torch.int32
            )
        else:
            polylines = torch.zeros(
                (1, num_points_polyline, 5), dtype=torch.float32
            )
            polylines_valid = torch.zeros((1,), dtype=torch.int32)

        if polylines.shape[0] >= max_polylines:
            polylines = polylines[:max_polylines]
            polylines_valid = polylines_valid[:max_polylines]
        else:
            # Apply padding to polylines
            polylines = torch.nn.functional.pad(
                polylines,
                pad=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    max_polylines - polylines.shape[0],
                ),  # (depth, height, width) padding
            )

            # Apply padding to polylines_valid
            polylines_valid = torch.nn.functional.pad(
                polylines_valid,
                pad=(
                    0,
                    max_polylines - polylines_valid.shape[0],
                ),  # width padding only
            )

    return polylines, polylines_valid


def sample_to_action():
    """Todo: Implement this function."""
    pass
