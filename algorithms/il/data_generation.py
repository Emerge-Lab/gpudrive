"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
import logging
import argparse
import itertools
import matplotlib.pyplot as plt
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Select the dynamics model that you use")
    parser.add_argument(
        "--dynamics-model",
        "-d",
        type=str,
        default="delta_local",
        choices=["delta_local", "bicycle", "classic"],
    )
    args = parser.parse_args()
    return args


def map_to_closest_discrete_value(grid, cont_actions):
    """
    Find the nearest value in the action grid for a given expert action.
    """
    # Calculate the absolute differences and find the indices of the minimum values
    abs_diff = torch.abs(grid.unsqueeze(0) - cont_actions.unsqueeze(-1))
    indx = torch.argmin(abs_diff, dim=-1)

    # Gather the closest values based on the indices
    closest_values = grid[indx]

    return closest_values, indx


def generate_state_action_pairs(
    env,
    device,
    action_space_type="discrete",
    use_action_indices=False,
    make_video=False,
    render_index=[0],
    num_action=10000,
    save_path="output_video.mp4",
    debug_world_idx=None,
    debug_veh_idx=None,
):
    """Generate pairs of states and actions from the Waymo Open Dataset.

    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
        device (str): Where to run the simulation (cpu or cuda).
        action_space_type (str): discrete, multi-discrete, continuous
        use_action_indices (bool): Whether to return action indices instead of action values.
        make_video (bool): Whether to save a video of the expert trajectory.
        render_index (int): Index of the world to render (must be <= num_worlds).

    Returns:
        expert_actions: Expert actions for the controlled agents. An action is a
            tuple with (acceleration, steering, heading).
        obs_tensor: Expert observations for the controlled agents.
    """
    frames = [[] for _ in range(render_index[1] - render_index[0])]

    logging.info(
        f"Generating expert actions and observations for {env.num_worlds} worlds \n"
    )

    # Reset the environment
    obs = env.reset()

    # Get expert actions for full trajectory in all worlds
    expert_actions, expert_speeds, expert_positions = env.get_expert_actions(
        debug_world_idx, debug_veh_idx
    )
    raw_expert_action = expert_actions.clone()

    expert_dx, expert_dy, expert_dyaw = None, None, None
    expert_accel, expert_steer = None, None
    if debug_world_idx is not None and debug_veh_idx is not None:
        if env.config.dynamics_model == "delta_local":
            expert_dx = raw_expert_action[debug_world_idx, debug_veh_idx, :, 0]
            expert_dy = raw_expert_action[debug_world_idx, debug_veh_idx, :, 1]
            expert_dyaw = raw_expert_action[
                debug_world_idx, debug_veh_idx, :, 2
            ]
        else:
            expert_accel = raw_expert_action[
                debug_world_idx, debug_veh_idx, :, 0
            ]
            expert_steer = raw_expert_action[
                debug_world_idx, debug_veh_idx, :, 1
            ]
        # for (pos_x, pos_y), speed in zip(expert_positions, expert_speeds):
        #     print(f'position : ({pos_x}. {pos_y}), speed : {speed}')
    if action_space_type == "discrete":
        logging.info("Discretizing expert actions... \n")
        # Discretize the expert actions: map every value to the closest
        # value in the action grid.
        disc_expert_actions = expert_actions.clone()
        if env.config.dynamics_model == "delta_local":
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.dx, cont_actions=expert_actions[:, :, :, 0]
            )
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.dy, cont_actions=expert_actions[:, :, :, 1]
            )
            disc_expert_actions[:, :, :, 2], _ = map_to_closest_discrete_value(
                grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2]
            )
        else:
            # Acceleration
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.accel_actions, cont_actions=expert_actions[:, :, :, 0]
            )
            # Steering
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.steer_actions, cont_actions=expert_actions[:, :, :, 1]
            )

        if use_action_indices:  # Map action values to joint action index
            logging.info("Mapping expert actions to joint action index... \n")
            expert_action_indices = torch.zeros(
                expert_actions.shape[0],
                expert_actions.shape[1],
                expert_actions.shape[2],
                1,
                dtype=torch.int32,
            ).to(device)
            for world_idx in range(disc_expert_actions.shape[0]):
                for agent_idx in range(disc_expert_actions.shape[1]):
                    for time_idx in range(disc_expert_actions.shape[2]):
                        action_val_tuple = tuple(
                            round(x, 3)
                            for x in disc_expert_actions[
                                world_idx, agent_idx, time_idx, :
                            ].tolist()
                        )
                        if not env.config.dynamics_model == "delta_local":
                            action_val_tuple = (
                                action_val_tuple[0],
                                action_val_tuple[1],
                                0.0,
                            )

                        action_idx = env.values_to_action_key.get(
                            action_val_tuple
                        )
                        expert_action_indices[
                            world_idx, agent_idx, time_idx
                        ] = action_idx

            expert_actions = expert_action_indices
        else:
            # Map action values to joint action index
            expert_actions = disc_expert_actions
    elif action_space_type == "multi_discrete":
        """will be update"""
        pass
    else:
        logging.info("Using continuous expert actions... \n")

    # Storage
    expert_observations_lst = []
    expert_actions_lst = []
    expert_next_obs_lst = []
    expert_dones_lst = []

    # Initialize dead agent mask

    dead_agent_mask = ~env.cont_agent_mask.clone()
    alive_agent_mask = env.cont_agent_mask.clone()
    if debug_world_idx is not None and debug_veh_idx is not None:
        speeds = [obs[debug_world_idx, debug_veh_idx, 0].unsqueeze(-1)]
        poss = [obs[debug_world_idx, debug_veh_idx, 3:5].unsqueeze(0)]

    for time_step in range(env.episode_len):

        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])

        next_obs = env.get_obs()

        dones = env.get_dones()
        infos = env.get_infos()

        if debug_world_idx is not None and debug_veh_idx is not None:
            if dones[debug_world_idx, debug_veh_idx] == 0:
                speeds.append(
                    next_obs[debug_world_idx, debug_veh_idx, 0].unsqueeze(-1)
                )
                poss.append(
                    next_obs[debug_world_idx, debug_veh_idx, 3:5].unsqueeze(0)
                )

        # Unpack and store (obs, action, next_obs, dones) pairs for controlled agents
        expert_observations_lst.append(obs[~dead_agent_mask, :])
        expert_actions_lst.append(
            expert_actions[~dead_agent_mask][:, time_step, :]
        )

        expert_next_obs_lst.append(next_obs[~dead_agent_mask, :])
        expert_dones_lst.append(dones[~dead_agent_mask])

        # Update
        obs = next_obs
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        # Render
        if make_video:
            for render in range(render_index[0], render_index[1]):
                frame = env.render(world_render_idx=render)
                frames[render].append(frame)
        if (dead_agent_mask == True).all():
            break

    is_collision = infos[:, :, :3].sum(dim=-1)
    is_goal = infos[:, :, 3]
    collision_mask = is_collision != 0
    goal_mask = is_goal != 0
    valid_collision_mask = collision_mask & alive_agent_mask
    valid_goal_mask = goal_mask & alive_agent_mask
    collision_rate = (
        valid_collision_mask.sum().float() / alive_agent_mask.sum().float()
    )
    goal_rate = valid_goal_mask.sum().float() / alive_agent_mask.sum().float()

    # print(f'Action: dx {combi[0]} dy {combi[1]} dyaw {combi[2]} | Collision {collision_rate} Goal {goal_rate}')

    if debug_world_idx is not None:
        speeds = torch.cat(speeds)
        print(f"Environment speeds {speeds}")
        poss = torch.cat(poss, dim=0)
        print(f"Environment poss {poss}")
    if make_video:
        for render in range(render_index[0], render_index[1]):
            imageio.mimwrite(
                f"{save_path}_world_{render}.mp4",
                np.array(frames[render]),
                fps=30,
            )

    flat_expert_obs = torch.cat(expert_observations_lst, dim=0)
    flat_expert_actions = torch.cat(expert_actions_lst, dim=0)
    flat_next_expert_obs = torch.cat(expert_next_obs_lst, dim=0)
    flat_expert_dones = torch.cat(expert_dones_lst, dim=0)
    if debug_world_idx is not None:
        """for plotting"""
        if env.config.dynamics_model == "delta_local":
            fig, axs = plt.subplots(2, 3, figsize=(8, 8))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        # Speed plot
        axs[0, 0].plot(expert_speeds.numpy(), label="Expert Speeds", color="b")
        axs[0, 0].plot(speeds.numpy(), label="Simulation Speeds", color="r")
        axs[0, 0].set_title("Speeds Comparison")
        axs[0, 0].set_xlabel("Time Step")
        axs[0, 0].set_ylabel("Speed")
        axs[0, 0].legend()

        # Position plot
        axs[0, 1].plot(
            expert_positions[:, 1].numpy(),
            expert_positions[:, 0].numpy(),
            label="Expert Position",
            color="b",
            marker="o",
        )
        axs[0, 1].plot(
            poss[:, 0].numpy(),
            poss[:, 1].numpy(),
            label="Environment Position",
            color="r",
            marker="x",
        )
        axs[0, 1].set_title("Position Comparison with Order")
        axs[0, 1].set_xlabel("X Position")
        axs[0, 1].set_ylabel("Y Position")
        axs[0, 1].legend()

        if env.config.dynamics_model == "delta_local":
            # dx plot
            axs[1, 0].plot(expert_dx.numpy(), label="Expert dx", color="b")
            axs[1, 0].plot(
                disc_expert_actions[
                    debug_world_idx, debug_veh_idx, :, 0
                ].numpy(),
                label="Simulation dx",
                color="r",
            )
            axs[1, 0].set_title("dx Comparison")
            axs[1, 0].set_xlabel("Time Step")
            axs[1, 0].set_ylabel("dx")
            axs[1, 0].legend()

            # dy plot
            axs[1, 1].plot(expert_dy.numpy(), label="Expert dy", color="b")
            axs[1, 1].plot(
                disc_expert_actions[
                    debug_world_idx, debug_veh_idx, :, 1
                ].numpy(),
                label="Simulation dy",
                color="r",
            )
            axs[1, 1].set_title("dy Comparison")
            axs[1, 1].set_xlabel("Time Step")
            axs[1, 1].set_ylabel("dy")
            axs[1, 1].legend()

            # dyaw plot
            axs[1, 2].plot(expert_dyaw.numpy(), label="Expert dyaw", color="b")
            axs[1, 2].plot(
                disc_expert_actions[
                    debug_world_idx, debug_veh_idx, :, 2
                ].numpy(),
                label="Simulation dyaw",
                color="r",
            )
            axs[1, 2].set_title("dyaw Comparison")
            axs[1, 2].set_xlabel("Time Step")
            axs[1, 2].set_ylabel("dyaw")
            axs[1, 2].legend()
        else:
            # Accels plot
            axs[1, 0].plot(
                expert_accel.numpy(), label="Expert Accels", color="b"
            )
            axs[1, 0].plot(
                disc_expert_actions[
                    debug_world_idx, debug_veh_idx, :, 0
                ].numpy(),
                label="Simulation Accels",
                color="r",
            )
            axs[1, 0].set_title("Accels Comparison")
            axs[1, 0].set_xlabel("Time Step")
            axs[1, 0].set_ylabel("Accels")
            axs[1, 0].legend()

            # Steers plot
            axs[1, 1].plot(
                expert_steer.numpy(), label="Expert Steers", color="b"
            )
            axs[1, 1].plot(
                disc_expert_actions[
                    debug_world_idx, debug_veh_idx, :, 1
                ].numpy(),
                label="Simulation Steers",
                color="r",
            )
            axs[1, 1].set_title("Steers Comparison")
            axs[1, 1].set_xlabel("Time Step")
            axs[1, 1].set_ylabel("Steers")
            axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(
            f"Analysis_numAction{num_action}_W{debug_world_idx}_V{debug_veh_idx}.jpg",
            dpi=150,
        )
    return (
        flat_expert_obs,
        flat_expert_actions,
        flat_next_expert_obs,
        flat_expert_dones,
        goal_rate,
        collision_rate,
    )


if __name__ == "__main__":
    import argparse

    args = parse_args()
    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = 10
    MAX_NUM_OBJECTS = 128

    # Initialize lists to store results
    num_actions = []
    goal_rates = []
    collision_rates = []

    # Set the environment and render configurations
    # Action space (joint discrete)
    num_dx_values = reversed(range(100, 101, 1))
    num_dy_values = reversed(range(100, 101, 1))
    num_dyaw_values = reversed(range(300, 301, 1))

    combinations = itertools.product(
        num_dx_values, num_dy_values, num_dyaw_values
    )
    render_config = RenderConfig(draw_obj_idx=True)
    scene_config = SceneConfig(
        "/data/formatted_json_v2_no_tl_train/", NUM_WORLDS
    )
    for combi in combinations:
        num_dx, num_dy, num_dyaw = combi
        env_config = EnvConfig(
            dynamics_model=args.dynamics_model,
            steer_actions=torch.round(
                torch.linspace(-0.3, 0.3, 7), decimals=3
            ),
            accel_actions=torch.round(
                torch.linspace(-6.0, 6.0, 7), decimals=3
            ),
            dx=torch.round(torch.linspace(-3.0, 3.0, num_dx), decimals=3),
            dy=torch.round(torch.linspace(-3.0, 3.0, num_dy), decimals=3),
            dyaw=torch.round(torch.linspace(-1.0, 1.0, num_dyaw), decimals=3),
        )

        env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
            device="cpu",
            render_config=render_config,
        )
        # Generate expert actions and observations
        (
            expert_obs,
            expert_actions,
            next_expert_obs,
            expert_dones,
            goal_rate,
            collision_rate,
        ) = generate_state_action_pairs(
            env=env,
            device="cpu",
            action_space_type="discrete",  # Discretize the expert actions
            use_action_indices=True,  # Map action values to joint action index
            make_video=True,  # Record the trajectories as sanity check
            render_index=[0, 0],  # start_idx, end_idx
            debug_world_idx=None,
            debug_veh_idx=None,
            save_path="use_discr_actions_fix",
            num_action=combi,
        )
        env.close()
        del env
        del env_config
        # Calculate the total number of actions based on the combination
        num_action = num_dx * num_dy * num_dyaw

        # Store the results
        num_actions.append(num_action)
        goal_rates.append(goal_rate.cpu().numpy())
        collision_rates.append(collision_rate.cpu().numpy())
        print(
            f"dx {num_dx} dy {num_dy} dyaw {num_dyaw} Collision rate {collision_rate} Goal RATE {goal_rate}"
        )
        # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(num_actions, goal_rates, label="Goal Rate", marker="o")
    plt.plot(num_actions, collision_rates, label="Collision Rate", marker="x")
    plt.xlabel("Number of Actions")
    plt.ylabel("Rate")
    plt.title("Goal Rate and Collision Rate vs. Number of Actions")
    plt.legend()
    plt.savefig("Trade-off.jpg", dpi=300)
    # Uncommment to save the expert actions and observations
    # torch.save(expert_actions, "expert_actions.pt")
    # torch.save(expert_obs, "expert_obs.pt")
