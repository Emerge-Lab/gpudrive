"""Functions used for processing roadgraph data and other features for VBD."""
import torch

from vbd.data.data_utils import calculate_relations

def process_scenario_data(
    num_envs,
    max_controlled_agents,
    controlled_agent_mask,
    global_agent_observations,
    road_graph,
    init_steps,
    episode_len,
    positions,
    velocities,
    yaws,
):
    
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
    (num_envs, max_controlled_agents, episode_len - init_steps, 5))
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
        road_graph, num_envs, agents_history, agents_interested
    )

    # Empty (16, 3)
    traffic_light_points = torch.zeros((1, 16, 3))

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
        "polylines": polylines,
        "polylines_valid": polylines_valid,
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
            global_traj[:, :, 10].unsqueeze(-1).expand(-1, -1, init_steps + 1).unsqueeze(-1),  # vehicle len
            global_traj[:, :, 11].unsqueeze(-1).expand(-1, -1, init_steps + 1).unsqueeze(-1),  # vehicle width
            torch.ones(
                (num_envs, max_cont_agents, init_steps + 1)
            ).unsqueeze(
                -1
            ),  # vehicle height
        ],
        dim=-1,
    )

    agents_history[~controlled_agent_mask, :, :] = 0.0

    return agents_history


def construct_polylines(road_graph, num_envs, agents_history, agents_interested, max_polylines=30, num_road_points=256):
    """Get the global polylines information."""

    # Have: p_x, p_y, scale, heading, type
    # TODO: Need: p_x, p_y, heading, traffic_light_state, lane_type
    # TODO: Goal shape (256, 30, 5)
    # (roadgraph_top_k, 7)
    ### Convert type to waymax type
    lane_types = road_graph[:, :max_polylines*num_road_points, 6].long()

    polylines = torch.cat(
        [
            road_graph[:, :max_polylines*num_road_points, :2],  # x, y (3D tensor)
            road_graph[:, :max_polylines*num_road_points, 5:6],  # heading (unsqueezed to 3D)
            torch.zeros_like(
                road_graph[:, :max_polylines*num_road_points, 5:6]
            ),  # traffic_light_state (unsqueezed to 3D)
            lane_types.long().unsqueeze(
                -1
            ),  # lane_type (unsqueezed to 3D)
        ],
        dim=-1,  # Concatenate along the last dimension
    )

    # TODO(dc): Process the polylines tensor using the road IDs
    polylines = polylines.reshape(num_road_points[:, :max_polylines*num_road_points, :], max_polylines, 5)

    # TODO(dc): Create correct valid tensor
    polylines_valid = torch.ones((num_envs, 10))

    return polylines, polylines_valid