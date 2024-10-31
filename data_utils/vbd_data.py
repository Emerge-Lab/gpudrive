"""Functions used for processing roadgraph data and other features for VBD."""
import torch

from vbd.data.data_utils import calculate_relations


def process_scenario_data(
    num_envs,
    max_controlled_agents,
    controlled_agent_mask,
    global_agent_observations,
    global_road_graph,
    local_road_graph,
    init_steps,
    episode_len,
    positions,
    velocities,
    yaws,
    max_polylines=256,
    num_points_polyline=30,
):
    """Process the scenario data for Versatile Behavior Diffusion."""

    agents_history = construct_agent_history(
        init_steps=init_steps,
        controlled_agent_mask=controlled_agent_mask,
        max_cont_agents=max_controlled_agents,
        num_envs=num_envs,
        global_agent_observations=global_agent_observations,
        pos_xy=positions,
        vel_xy=velocities,
        yaw=yaws,
    )

    # Now create the agents future logs
    agents_future = torch.zeros(
        (num_envs, max_controlled_agents, episode_len - init_steps, 5)
    )
    agents_future[:, :, :, 0:2] = positions[:, :, init_steps:, :]
    agents_future[:, :, :, 2] = yaws[:, :, init_steps:, :].squeeze(-1)
    agents_future[:, :, :, 3:5] = velocities[:, :, init_steps:, :]

    # Set all invalid agent values to zero
    agents_future[~controlled_agent_mask, :, :] = 0

    # Currently, all agents are vehicles, encoding as type 1
    agents_type = torch.zeros([num_envs, max_controlled_agents])
    agents_type[controlled_agent_mask] = 1

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
        "anchors": torch.zeros((1, 32, 64, 2)),  # Placeholder
    }

    return data_dict


def construct_agent_history(
    init_steps,
    controlled_agent_mask,
    max_cont_agents,
    num_envs,
    global_agent_observations,
    pos_xy,
    vel_xy,
    yaw,
):
    """Get the agent trajectory feature information."""

    global_traj = global_agent_observations
    global_traj[~controlled_agent_mask] = 0.0
    global_traj = global_traj[:, :max_cont_agents, :]

    # x, y, heading, vel_x, vel_y, len, width, height
    agents_history = torch.cat(
        [
            pos_xy[:, :, : init_steps + 1, :],  # x, y
            yaw[:, :, : init_steps + 1, :],  # heading
            vel_xy[:, :, : init_steps + 1, :],  # vel_x, vel_y
            global_traj[:, :, 10]
            .unsqueeze(-1)
            .expand(-1, -1, init_steps + 1)
            .unsqueeze(-1),  # vehicle len
            global_traj[:, :, 11]
            .unsqueeze(-1)
            .expand(-1, -1, init_steps + 1)
            .unsqueeze(-1),  # vehicle width
            torch.ones((num_envs, max_cont_agents, init_steps + 1)).unsqueeze(
                -1
            ),  # vehicle height
        ],
        dim=-1,
    )

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
    # TODO(dc): Vectorize for multiple environments
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
