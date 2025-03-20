import torch
import pandas as pd
import numpy as np

from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState

import torch
import pandas as pd
import numpy as np

from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState


def rollout(
    env,
    policy,
    device,
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 1,
    zoom_radius: int = 100,
    return_agent_positions: bool = False,
    center_on_ego: bool = False,
    set_agent_type: bool = False,
    agent_weights: torch.Tensor = None,
):
    """
    Perform a rollout of a policy in the environment.

    Args:
        env: The simulation environment.
        policy: The policy to be rolled out.
        device: The device to execute computations on (CPU/GPU).
        deterministic (bool): Whether to use deterministic policy actions.
        render_sim_state (bool): Whether to render the simulation state.
        render_every_n_steps (int): Render every N steps.
        zoom_radius (int): Radius for zoom in visualization.
        return_agent_positions (bool): Whether to return agent positions.
        center_on_ego (bool): Whether to center visualization on ego vehicle.
        set_agent_type (bool): Whether to set agent type during reset.
        agent_weights (torch.Tensor): Agent weights tensor for condition_mode="fixed".

    Returns:
        tuple: Averages for goal achieved, collisions, off-road occurrences,
               controlled agents count, and simulation state frames.
    """
    # Initialize storage
    sim_state_frames = {env_id: [] for env_id in range(env.num_worlds)}
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len
    agent_positions = torch.zeros(
        (env.num_worlds, env.max_agent_count, episode_len, 2)
    )

    # Reset episode
    if set_agent_type and agent_weights is not None:
        # Pass agent_weights to reset when set_agent_type is True
        next_obs = env.reset(condition_mode="fixed", agent_type=agent_weights)
    else:
        next_obs = env.reset()

    # Storage
    goal_achieved = torch.zeros((num_worlds, max_agent_count), device=device)
    collided = torch.zeros((num_worlds, max_agent_count), device=device)
    off_road = torch.zeros((num_worlds, max_agent_count), device=device)
    active_worlds = np.arange(num_worlds).tolist()
    episode_lengths = torch.zeros(num_worlds)

    control_mask = env.cont_agent_mask
    live_agent_mask = control_mask.clone()

    for time_step in range(episode_len):

        # Get actions for active agents
        if live_agent_mask.any():
            action, _, _, _ = policy(
                next_obs[live_agent_mask], deterministic=deterministic
            )

            # Insert actions into a template
            action_template = torch.zeros(
                (num_worlds, max_agent_count), dtype=torch.int64, device=device
            )
            action_template[live_agent_mask] = action.to(device)

            # Step the environment
            env.step_dynamics(action_template)

            # Render
            if render_sim_state and len(active_worlds) > 0:

                has_live_agent = torch.where(
                    live_agent_mask[active_worlds, :].sum(axis=1) > 0
                )[0].tolist()

                if time_step % render_every_n_steps == 0:
                    if center_on_ego:
                        agent_indices = torch.argmax(
                            control_mask.to(torch.uint8), dim=1
                        ).tolist()
                    else:
                        agent_indices = None

                    sim_state_figures = env.vis.plot_simulator_state(
                        env_indices=has_live_agent,
                        time_steps=[time_step] * len(has_live_agent),
                        zoom_radius=zoom_radius,
                        center_agent_indices=agent_indices,
                    )
                    for idx, env_id in enumerate(has_live_agent):
                        sim_state_frames[env_id].append(
                            img_from_fig(sim_state_figures[idx])
                        )

        # Update observations, dones, and infos
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()

        off_road[live_agent_mask] += infos.off_road[live_agent_mask]
        collided[live_agent_mask] += infos.collided[live_agent_mask]
        goal_achieved[live_agent_mask] += infos.goal_achieved[live_agent_mask]

        # Update live agent mask
        live_agent_mask[dones] = False

        # Process completed worlds
        num_dones_per_world = (dones & control_mask).sum(dim=1)
        total_controlled_agents = control_mask.sum(dim=1)
        done_worlds = (num_dones_per_world == total_controlled_agents).nonzero(
            as_tuple=True
        )[0]

        for world in done_worlds:
            if world in active_worlds:
                active_worlds.remove(world)
                episode_lengths[world] = time_step

        if return_agent_positions:
            global_agent_states = GlobalEgoState.from_tensor(
                env.sim.absolute_self_observation_tensor()
            )
            agent_positions[:, :, time_step, 0] = global_agent_states.pos_x
            agent_positions[:, :, time_step, 1] = global_agent_states.pos_y

        if not active_worlds:  # Exit early if all worlds are done
            break

    # Aggregate metrics to obtain averages across scenes
    controlled_per_scene = control_mask.sum(dim=1).float()

    # Counts
    goal_achieved_count = (goal_achieved > 0).float().sum(axis=1)
    collided_count = (collided > 0).float().sum(axis=1)
    off_road_count = (off_road > 0).float().sum(axis=1)
    not_goal_nor_crash_count = (
        torch.logical_and(
            goal_achieved == 0,  # Didn't reach the goal
            torch.logical_and(
                collided == 0,  # Didn't collide
                torch.logical_and(
                    off_road == 0,  # Didn't go off-road
                    control_mask,  # Only count controlled agents
                ),
            ),
        )
        .float()
        .sum(dim=1)
    )

    # Fractions per scene
    frac_goal_achieved = goal_achieved_count / controlled_per_scene
    frac_collided = collided_count / controlled_per_scene
    frac_off_road = off_road_count / controlled_per_scene
    frac_not_goal_nor_crash_per_scene = (
        not_goal_nor_crash_count / controlled_per_scene
    )

    return (
        goal_achieved_count,
        frac_goal_achieved,
        collided_count,
        frac_collided,
        off_road_count,
        frac_off_road,
        not_goal_nor_crash_count,
        frac_not_goal_nor_crash_per_scene,
        controlled_per_scene,
        sim_state_frames,
        agent_positions,
        episode_lengths,
    )


def multi_policy_rollout(
    env,
    policies,
    device,
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 1,
    zoom_radius: int = 100,
    return_agent_positions: bool = False,
    center_on_ego: bool = False,
):
    """
    Perform a rollout of multiple policies in the environment.

    Args:
        env: The simulation environment.
        policies (dict): Dictionary of policies {policy_name: (policy_function,mask)}.
        device: The device to execute computations on (CPU/GPU).
        policy_masks (dict): Dictionary of policy masks {policy_name: mask_tensor}.
        deterministic (bool): Whether to use deterministic policy actions.
        return_agent_positions (bool): Whether to return agent positions.

    Returns:
        policy_metrics: Dictionary of metrics corresponding to policies {policy_name: metrics(dict)}
            metrics: {
                'goal_achieved', 'collided', 'off_road', 'off_road_count', 'collided_count', 'goal_achieved_count', 'frac_off_road', 'frac_collided', 'frac_goal_achieved'
                }

    """

    # Initialize storage
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len
    sim_state_frames = {env_id: [] for env_id in range(num_worlds)}
    agent_positions = torch.zeros(
        (num_worlds, max_agent_count, episode_len, 2)
    )

    # Reset environment
    next_obs = env.reset()
    policy_metrics = {
        policy_name: {
            "goal_achieved": torch.zeros(
                (num_worlds, max_agent_count), device=device
            ),
            "collided": torch.zeros(
                (num_worlds, max_agent_count), device=device
            ),
            "off_road": torch.zeros(
                (num_worlds, max_agent_count), device=device
            ),
        }
        for policy_name in policies
    }
    episode_lengths = torch.zeros(num_worlds)

    active_worlds = list(range(num_worlds))
    control_mask = env.cont_agent_mask
    live_agent_mask = control_mask.clone()

    for time_step in range(episode_len):

        policy_live_masks = {
            name: mask & live_agent_mask
            for name, (policy_fn, mask) in policies.items()
        }

        actions = {}
        for policy_name, (policy_fn, policy_mask) in policies.items():
            live_mask = policy_live_masks[policy_name]
            if live_mask.any():
                actions[policy_name], _, _, _ = policy_fn(
                    next_obs[live_mask], deterministic=deterministic
                )

        combined_mask = torch.zeros_like(live_agent_mask, dtype=torch.bool)
        for live_mask in policy_live_masks.values():
            combined_mask |= live_mask
        assert torch.all(
            live_agent_mask == combined_mask
        ), "Live agent mask mismatch!"

        action_template = torch.zeros(
            (num_worlds, max_agent_count), dtype=torch.int64, device=device
        )

        # Assign actions based on policy masks
        for policy_name, action in actions.items():
            live_mask = policy_live_masks[policy_name]
            if action.numel() > 0:
                action_template[live_mask] = action.to(
                    dtype=action_template.dtype, device=device
                )

        # Step environment
        env.step_dynamics(action_template)

        if render_sim_state and len(active_worlds) > 0:

            has_live_agent = torch.where(
                live_agent_mask[active_worlds, :].sum(axis=1) > 0
            )[0].tolist()

            if time_step % render_every_n_steps == 0:
                if center_on_ego:
                    agent_indices = torch.argmax(
                        control_mask.to(torch.uint8), dim=1
                    ).tolist()
                else:
                    agent_indices = None

                sim_state_figures = env.vis.plot_simulator_state(
                    env_indices=has_live_agent,
                    time_steps=[time_step] * len(has_live_agent),
                    zoom_radius=zoom_radius,
                    center_agent_indices=agent_indices,
                    policy_masks=policies,
                )
                for idx, env_id in enumerate(has_live_agent):
                    sim_state_frames[env_id].append(
                        img_from_fig(sim_state_figures[idx])
                    )

        # Update observations and agent statuses
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()

        for policy_name, live_mask in policy_live_masks.items():
            policy_metrics[policy_name]["off_road"][
                live_mask
            ] += infos.off_road[live_mask]
            policy_metrics[policy_name]["collided"][
                live_mask
            ] += infos.collided[live_mask]
            policy_metrics[policy_name]["goal_achieved"][
                live_mask
            ] += infos.goal_achieved[live_mask]

        live_agent_mask[dones] = False

        # Process completed worlds
        num_dones_per_world = (dones & control_mask).sum(dim=1)
        total_controlled_agents = control_mask.sum(dim=1)
        done_worlds = (num_dones_per_world == total_controlled_agents).nonzero(
            as_tuple=True
        )[0]

        for world in done_worlds:
            if world in active_worlds:
                active_worlds.remove(world)
                episode_lengths[world] = time_step

        if return_agent_positions:
            global_agent_states = GlobalEgoState.from_tensor(
                env.sim.absolute_self_observation_tensor()
            )
            agent_positions[:, :, time_step, 0] = global_agent_states.pos_x
            agent_positions[:, :, time_step, 1] = global_agent_states.pos_y

        if not active_worlds:
            break

    controlled_per_scene = sum(
        mask.sum(dim=1).float()
        for policy_name, (policy_fn, mask) in policies.items()
    )

    metrics = compute_metrics(
        policy_metrics, policy_live_masks, controlled_per_scene
    )

    if render_sim_state:
        return metrics, sim_state_frames

    return metrics


def compute_metrics(policy_metrics, policy_live_masks, controlled_per_scene):

    for policy_name, live_mask in policy_live_masks.items():

        policy_metrics[policy_name]["off_road_count"] = (
            (policy_metrics[policy_name]["off_road"] > 0).float().sum(axis=1)
        )
        policy_metrics[policy_name]["collided_count"] = (
            (policy_metrics[policy_name]["collided"] > 0).float().sum(axis=1)
        )
        policy_metrics[policy_name]["goal_achieved_count"] = (
            (policy_metrics[policy_name]["goal_achieved"] > 0)
            .float()
            .sum(axis=1)
        )

        policy_metrics[policy_name]["frac_off_road"] = (
            policy_metrics[policy_name]["off_road_count"]
            / controlled_per_scene
        )
        policy_metrics[policy_name]["frac_collided"] = (
            policy_metrics[policy_name]["collided_count"]
            / controlled_per_scene
        )
        policy_metrics[policy_name]["frac_goal_achieved"] = (
            policy_metrics[policy_name]["goal_achieved_count"]
            / controlled_per_scene
        )

    return policy_metrics


def create_data_table(data):
    # Extract unique policies
    policies = sorted(set(policy for pair in data.keys() for policy in pair))

    # Create empty DataFrames
    collisions_table = pd.DataFrame(index=policies, columns=policies)
    off_roads_table = pd.DataFrame(index=policies, columns=policies)
    goal_achieved_table = pd.DataFrame(index=policies, columns=policies)

    # Populate DataFrames
    for (p1, p2), metrics in data.items():
        collisions_table.loc[p1, p2] = metrics["frac_collided"].item()
        off_roads_table.loc[p1, p2] = metrics["frac_off_road"].item()
        goal_achieved_table.loc[p1, p2] = metrics["frac_goal_achieved"].item()

    # Print Tables
    print("Average Collisions Table:")
    print(collisions_table, "\n")

    print("Average Off Roads Table:")
    print(off_roads_table, "\n")

    print("Average Goal Achieved Table:")
    print(goal_achieved_table, "\n")
