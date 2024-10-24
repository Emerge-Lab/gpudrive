"""Functions used for processing roadgraph data and other features for VBD."""


def process_scenario():
    # Get the agents history
    agents_history = construct_agent_history(pos, vel, yaw, init_steps)

    # Now create the agents future logs
    agents_future[:, :, :, 0:2] = pos[:, :, init_steps:, :]
    agents_future[:, :, :, 2] = yaw[:, :, init_steps:, :].squeeze(-1)
    agents_future[:, :, :, 3:5] = vel[:, :, init_steps:, :]
    # Set all invalid agent values to zero
    agents_future[~self.cont_agent_mask, :, :] = 0

    # Currently, all agents are vehicles, encoding as type 1
    agents_type = torch.zeros([self.num_worlds, self.max_cont_agents])
    agents_type[self.cont_agent_mask] = 1

    # 10 if we are controlling the agent, 1 otherwise
    agents_interested = torch.ones([self.num_worlds, self.max_cont_agents])
    agents_interested[self.cont_agent_mask] = 10

    # Global polylines tensor: Shape (256, 30, 5)
    polylines, polylines_valid = self.construct_polylines(
        agents_history, agents_interested
    )

    # Empty (16, 3)
    traffic_light_points = torch.zeros((1, 16, 3))

    # Controlled agents
    agents_id = torch.nonzero(self.cont_agent_mask[0, :]).permute(1, 0)

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


def construct_polylines(self, agents_history, agents_interested):
    """Get the global polylines information."""

    # Automatically generate the waymax_types_to_gpudrive mapping using a dictionary comprehension
    waymax_types_to_gpudrive = {
        int(entity_type): gpudrive.mapRoadEntityTypeToID(entity_type)
        for entity_type in [
            gpudrive.EntityType._None,  # Using _None if None isn't allowed directly
            gpudrive.EntityType.RoadEdge,
            gpudrive.EntityType.RoadLine,
            gpudrive.EntityType.RoadLane,
            gpudrive.EntityType.CrossWalk,
            gpudrive.EntityType.SpeedBump,
            gpudrive.EntityType.StopSign,
        ]
    }
    # Have: p_x, p_y, scale, heading, type
    # Need: p_x, p_y, heading, traffic_light_state, lane_type
    global_road_graph = self.sim.map_observation_tensor().to_torch()
    # (roadgraph_top_k, 7)

    # polylines = torch.cat(
    #     [
    #         global_road_graph[:, :, :, :2],  # x, y (3D tensor)
    #         global_road_graph[:, :, :, 5:6],  # heading (unsqueezed to 3D)
    #         torch.zeros_like(
    #             global_road_graph[:, :, :, 5:6]
    #         ),  # traffic_light_state (unsqueezed to 3D)
    #         lane_types.long().unsqueeze(
    #             -1
    #         ),  # lane_type (unsqueezed to 3D)
    #     ],
    #     dim=-1,  # Concatenate along the last dimension
    # )

    # # TODO(dc): Find out 30 shape
    # polylines = polylines[:, :30, :]

    # # Condition to check if the value is an integer between 0 and 6
    # condition = (
    #     (polylines[:, :, :, 4] >= 0)
    #     & (polylines[:, :, :, 4] <= 6)
    #     & (polylines[:, :, :, 4] == polylines[:, :, :, 4].long().float())
    # )

    # TODO(dc): Create the new tensor, setting 1 where the condition is true, and 0 otherwise
    polylines_valid = torch.ones((self.num_worlds, num_road_points))

    ### Convert type to waymax type
    orig_lane_types = global_road_graph[:, :, :, 6].long()
    lane_types = torch.zeros_like(orig_lane_types)

    for old_id, new_id in waymax_types_to_gpudrive.items():
        lane_types[orig_lane_types == old_id] = new_id

    return polylines.permute(0, 2, 1, 3), polylines_valid


def construct_agent_history(self, pos, vel, yaw, init_steps):
    """Get the agent trajectory feature information."""
    global_traj = (
        self.sim.absolute_self_observation_tensor().to_torch().clone()
    )
    global_traj[~self.cont_agent_mask] = 0.0
    global_traj = global_traj[:, : self.max_cont_agents, :]

    # x, y, heading, vel_x, vel_y, len, width, height
    agents_history = torch.cat(
        [
            pos[:, :, : init_steps + 1, :],  # x, y
            yaw[:, :, : init_steps + 1, :],  # heading
            vel[:, :, : init_steps + 1, :],  # vel_x, vel_y
            global_traj[:, :, 10]
            .unsqueeze(-1)
            .expand(-1, -1, init_steps + 1)
            .unsqueeze(-1),  # vehicle len
            global_traj[:, :, 11]
            .unsqueeze(-1)
            .expand(-1, -1, init_steps + 1)
            .unsqueeze(-1),  # vehicle width
            torch.ones(
                (self.num_worlds, self.max_cont_agents, init_steps + 1)
            ).unsqueeze(
                -1
            ),  # vehicle height
        ],
        dim=-1,
    )

    agents_history[~self.cont_agent_mask, :, :] = 0.0

    return agents_history


# filter_topk_roadgraph_points
