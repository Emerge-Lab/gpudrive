"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
import logging

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

logging.getLogger(__name__)


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
    discretize_actions=False,
    use_action_indices=False,
    make_video=False,
    render_index=[0, 1],
    save_path="output_video.mp4",
    debug_world_idx=None,
    debug_veh_idx=None,
):
    """Generate pairs of states and actions from the Waymo Open Dataset.

    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
        device (str): Where to run the simulation (cpu or cuda).
        discretize_actions (bool): Whether to discretize the expert actions.
        use_action_indices (bool): Whether to return action indices instead of action values.
        make_video (bool): Whether to save a video of the expert trajectory.
        render_index (int): Index of the world to render (must be <= num_worlds).

    Returns:
        expert_actions: Expert actions for the controlled agents. An action is a
            tuple with (acceleration, steering, heading).
        obs_tensor: Expert observations for the controlled agents.
    """
    print(render_index)
    frames = [[] for _ in range(render_index[1] - render_index[0])]

    logging.info(
        f"Generating expert actions and observations for {env.num_worlds} worlds \n"
    )

    # Reset the environment
    obs = env.reset()

    # Get expert actions for full trajectory in all worlds
    print(f"debug activated {debug_world_idx} {debug_veh_idx}")
    expert_actions, expert_speeds, expert_positions = env.get_expert_actions(debug_world_idx, debug_veh_idx)
    raw_expert_action = expert_actions.clone()
    print(f"debug activated {debug_world_idx} {debug_veh_idx}")
    print(debug_world_idx)
    if debug_world_idx != None:
        expert_accel = raw_expert_action[debug_world_idx, debug_veh_idx, :, 0]
        expert_steer = raw_expert_action[debug_world_idx, debug_veh_idx, :, 1]
        print(f'expert speeds {expert_speeds}')
        print(f'expert pos {expert_positions}')
    if discretize_actions:
        logging.info("Discretizing expert actions... \n")
        # Discretize the expert actions: map every value to the closest
        # value in the action grid.
        disc_expert_actions = expert_actions.clone()
        # print(f'disc expert action {disc_expert_actions.shape}')
        # Acceleration
        disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
            grid=env.accel_actions, cont_actions=expert_actions[:, :, :, 0]
        )
        # Steering
        disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
            grid=env.steer_actions, cont_actions=expert_actions[:, :, :, 1]
        )
        # print('values_to_action_key', env.values_to_action_key)
        # print('action_key_to_values', env.action_key_to_values)
        # print(disc_expert_actions[:, :, :, 2].sum() > 10)
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
                        action_val_tuple = (action_val_tuple[0], action_val_tuple[1], 0.0)
                        # print(round(action_val_tuple[2], 3))
                        action_idx = env.values_to_action_key.get(
                            action_val_tuple
                        )
                        # if world_idx == 0 and agent_idx == 5:
                        #     print(f'[data_generation.py] Time {time_idx} [accel, steer] {raw_expert_action[world_idx, agent_idx, time_idx, :-1]}')
                        #     print(f'[data_generation.py] Time {time_idx} [action_val_tuple, action_idx] {action_val_tuple, action_idx}')
                        expert_action_indices[
                            world_idx, agent_idx, time_idx
                        ] = action_idx

            expert_actions = expert_action_indices
        else:
            # Map action values to joint action index
            expert_actions = disc_expert_actions
    else:
        logging.info("Using continuous expert actions... \n")

    # Storage
    expert_observations_lst = []
    expert_actions_lst = []
    expert_next_obs_lst = []
    expert_dones_lst = []

    # Initialize dead agent mask


    dead_agent_mask = ~env.cont_agent_mask.clone()
    # print(dead_agent_mask[2])
    # print('Accel', torch.cat([expert_actions[0, 5, :, 0].unsqueeze(-1), raw_expert_action[0, 5, :, 0].unsqueeze(-1),
    #                           disc_expert_actions[0, 5, :, 0].unsqueeze(-1)], dim=-1))
    # print('steer', torch.cat([expert_actions[0, 5, :, 0].unsqueeze(-1), raw_expert_action[0, 5, :, 1].unsqueeze(-1),
    #                           disc_expert_actions[0, 5, :, 1].unsqueeze(-1)], dim=-1))
    # print(torch.cat([expert_actions[0, 0, :, :], raw_expert_action[0, 0, :, :-1]], dim=-1))
    # print(f'Agent 5 speed {obs[0, 5, 0]}')
    if debug_world_idx != None:
        speeds = [obs[debug_world_idx, debug_veh_idx, 0].unsqueeze(-1)]
        poss = [obs[debug_world_idx, debug_veh_idx, 3:5].unsqueeze(0)]
    # print(f'cont agent mask {env.cont_agent_mask.clone()[0]}')
    # print(f'dead ones {dead_agent_mask[0]}')
    for time_step in range(env.episode_len):

        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])

        next_obs = env.get_obs()
        # print(f'Time {time_step} Agent 5 action {expert_actions[0, 5, time_step, :]}')
        # print(f'Time {time_step} Agent 5 speed {next_obs[0, 5, 0]}')
        # print(f'Time {time_step} Agent 5 pos {next_obs[0, 5, 3:5]}')
        dones = env.get_dones()
        print(f'Dead? {dones[debug_world_idx, debug_veh_idx] }')
        # print(f'Action {expert_actions}')
        # print(dones[0, 5])
        if debug_world_idx:
            if dones[debug_world_idx, debug_veh_idx] == 0:
                speeds.append(next_obs[debug_world_idx, debug_veh_idx, 0].unsqueeze(-1))
                poss.append(next_obs[debug_world_idx, debug_veh_idx, 3:5].unsqueeze(0))
        # Unpack and store (obs, action, next_obs, dones) pairs for controlled agents
        expert_observations_lst.append(obs[~dead_agent_mask, :])
        expert_actions_lst.append(
            expert_actions[~dead_agent_mask][:, time_step, :]
        )

        expert_next_obs_lst.append(next_obs[~dead_agent_mask, :])
        expert_dones_lst.append(dones[~dead_agent_mask])

        # Update
        obs = next_obs
        print(dones[0])
        print(f'Dead Agent mask {dead_agent_mask[0]}')
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        # Render
        if make_video:
            for render in range(render_index[0], render_index[1]):
                frame = env.render(world_render_idx=render)
                frames[render].append(frame)

        if (dead_agent_mask == True).all():
            break
    if debug_world_idx != None:
        speeds = torch.cat(speeds)
        print(f'Environment speeds {speeds}' )
        poss = torch.cat(poss, dim=0)
        print(f'Environment poss {poss}' )
    if make_video:
        for render in range(render_index[0], render_index[1]):
            imageio.mimwrite(f'{save_path}_world_{render}.mp4', np.array(frames[render]), fps=30)

    flat_expert_obs = torch.cat(expert_observations_lst, dim=0)
    flat_expert_actions = torch.cat(expert_actions_lst, dim=0)
    flat_next_expert_obs = torch.cat(expert_next_obs_lst, dim=0)
    flat_expert_dones = torch.cat(expert_dones_lst, dim=0)

    '''for plotting '''
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    if debug_world_idx != None:
        # Speed plot
        axs[0, 0].plot(expert_speeds.numpy(), label='Expert Speeds', color='b')
        axs[0, 0].plot(speeds.numpy(), label='Simulation Speeds', color='r')
        axs[0, 0].set_title('Speeds Comparison')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Speed')
        axs[0, 0].legend()

        # Position plot
        axs[0, 1].plot(expert_positions[:, 0].numpy(), expert_positions[:, 1].numpy(), label='Expert Position', color='b',
                       marker='o')
        axs[0, 1].plot(poss[:, 0].numpy(), poss[:, 1].numpy(), label='Environment Position', color='r', marker='x')
        axs[0, 1].set_title('Position Comparison with Order')
        axs[0, 1].set_xlabel('X Position')
        axs[0, 1].set_ylabel('Y Position')
        axs[0, 1].legend()

        # Accels plot
        axs[1, 0].plot(expert_accel.numpy(), label='Expert Accels', color='b')
        axs[1, 0].plot(disc_expert_actions[debug_world_idx, debug_veh_idx, :, 0].numpy(), label='Simulation Accels', color='r')
        axs[1, 0].set_title('Accels Comparison')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Accels')
        axs[1, 0].legend()

        # Steers plot
        axs[1, 1].plot(expert_steer.numpy(), label='Expert Steers', color='b')
        axs[1, 1].plot(disc_expert_actions[debug_world_idx, debug_veh_idx, :, 1].numpy(), label='Simulation Steers', color='r')
        axs[1, 1].set_title('Steers Comparison')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Steers')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig('Analysis.jpg', dpi=150)
    return (
        flat_expert_obs,
        flat_expert_actions,
        flat_next_expert_obs,
        flat_expert_dones,
    )


if __name__ == "__main__":

    torch.set_printoptions(precision=3,sci_mode=False)
    NUM_WORLDS = 20
    MAX_NUM_OBJECTS = 128

    # Set the environment and render configurations
    # Action space (joint discrete)
    env_config = EnvConfig(
        use_bicycle_model=True,
        steer_actions=torch.round(
        torch.linspace(-0.3, 0.3, 100), decimals=3
    ),
        accel_actions=torch.round(
        torch.linspace(-6.0, 6.0, 100), decimals=3
    ))
    render_config = RenderConfig(draw_obj_idx=True)
    scene_config = SceneConfig("data", NUM_WORLDS)
    
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
    ) = generate_state_action_pairs(
        env=env,
        device="cpu",
        discretize_actions=True,  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=True,  # Record the trajectories as sanity check
        render_index=[0, 1], #start_idx, end_idx
        debug_world_idx=0,
        debug_veh_idx=5,
        save_path="use_discr_actions_fix",
    )

    # Uncommment to save the expert actions and observations 
    # torch.save(expert_actions, "expert_actions.pt")
    # torch.save(expert_obs, "expert_obs.pt")
