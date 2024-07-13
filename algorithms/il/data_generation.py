"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
import logging

from pygpudrive.env.config import EnvConfig, RenderConfig
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
    render_index=0,
    save_path="output_video.mp4",
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
    frames = []

    logging.info(
        f"Generating expert actions and observations for {env.num_worlds} worlds \n"
    )

    # Reset the environment
    obs = env.reset()

    # Get expert actions for full trajectory in all worlds
    expert_actions = env.get_expert_actions()

    if discretize_actions:
        logging.info("Discretizing expert actions... \n")
        # Discretize the expert actions: map every value to the closest
        # value in the action grid.
        disc_expert_actions = expert_actions.clone()

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
    else:
        logging.info("Using continuous expert actions... \n")

    # Storage
    expert_observations_lst = []
    expert_actions_lst = []
    expert_next_obs_lst = []
    expert_dones_lst = []

    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone()

    for time_step in range(env.episode_len):

        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])

        next_obs = env.get_obs()
        dones = env.get_dones()

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
            frame = env.render(world_render_idx=render_index)
            frames.append(frame)

        if (dead_agent_mask == True).all():
            break

    if make_video:
        imageio.mimwrite(save_path, np.array(frames), fps=30)

    flat_expert_obs = torch.cat(expert_observations_lst, dim=0)
    flat_expert_actions = torch.cat(expert_actions_lst, dim=0)
    flat_next_expert_obs = torch.cat(expert_next_obs_lst, dim=0)
    flat_expert_dones = torch.cat(expert_dones_lst, dim=0)

    return (
        flat_expert_obs,
        flat_expert_actions,
        flat_next_expert_obs,
        flat_expert_dones,
    )


if __name__ == "__main__":

    # Set the environment and render configurations
    env_config = EnvConfig(use_bicycle_model=True)
    render_config = RenderConfig()

    env = GPUDriveTorchEnv(
        config=env_config,
        render_config=render_config,
        num_worlds=3,
        max_cont_agents=128,
        data_dir="example_data",
        device="cuda",
    )

    # Generate expert actions and observations
    (
        expert_obs,
        expert_actions,
        next_expert_obs,
        expert_dones,
    ) = generate_state_action_pairs(
        env=env,
        device="cuda",
        discretize_actions=True,  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=True,  # Record the trajectories as sanity check
        render_index=1,
        save_path="use_discr_actions_fix.mp4",
    )

    # Save the expert actions and observations
    # torch.save(expert_actions, "expert_actions.pt")
    # torch.save(expert_obs, "expert_obs.pt")
