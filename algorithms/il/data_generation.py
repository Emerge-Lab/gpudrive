"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
from pygpudrive.env.config import EnvConfig, RenderConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv


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
    env_config,
    render_config,
    max_num_objects,
    num_worlds,
    data_dir,
    device,
    discretize_actions=False,
    make_video=False,
    save_path="output_video.mp4",
):
    """Generate pairs of states and actions from the Waymo Open Dataset.

    Args:
        env_config (EnvConfig): Environment configurations.
        render_config (RenderConfig): Render configurations.
        max_num_objects (int): Maximum controllable agents per world.
        num_worlds (int): Number of worlds (envs).
        data_dir (str): Path to folder with scenarios (Waymo Open Motion Dataset)
        device (str): Where to run the simulation (cpu or cuda).
        discretize_actions (bool): Whether to discretize the expert actions.
        make_video (bool): Whether to save a video of the expert trajectory.

    Returns:
        expert_actions: Expert actions for the controlled agents. An action is a
            tuple with (acceleration, steering, heading).
        obs_tensor: Expert observations for the controlled agents.
    """
    frames = []

    # Make environment with chosen scenarios
    env = GPUDriveTorchEnv(
        config=env_config,
        render_config=render_config,
        num_worlds=num_worlds,
        max_cont_agents=max_num_objects,
        data_dir=data_dir,
        device=device,
    )

    # Reset the environment
    obs = env.reset()

    # Get expert actions for full trajectory in all worlds
    expert_actions = env.get_expert_actions()
    if discretize_actions:
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
        expert_actions = disc_expert_actions

    # Storage
    expert_observations_lst = []
    expert_actions_lst = []

    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone()

    for time_step in range(env.episode_len):

        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])

        next_obs = env.get_obs()
        dones = env.get_dones()

        # Unpack and store (obs, action) pairs for controlled agents
        expert_observations_lst.append(obs[~dead_agent_mask, :])
        expert_actions_lst.append(
            expert_actions[~dead_agent_mask][:, time_step, :]
        )

        # Update
        obs = next_obs
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        # Render
        if make_video:
            frame = env.render(world_render_idx=2)
            frames.append(frame)

        if (dead_agent_mask == True).all():
            break

    if make_video:
        imageio.mimwrite(save_path, np.array(frames), fps=30)

    flattened_expert_obs = torch.cat(expert_observations_lst, dim=0)
    flattened_expert_actions = torch.cat(expert_actions_lst, dim=0)

    return flattened_expert_obs, flattened_expert_actions


if __name__ == "__main__":

    # Set the environment and render configurations
    env_config = EnvConfig(sample_method="pad_n", use_bicycle_model=True)
    render_config = RenderConfig()

    # Generate expert actions and observations
    expert_obs, expert_actions = generate_state_action_pairs(
        env_config=env_config,
        render_config=render_config,
        max_num_objects=128,
        num_worlds=10,
        data_dir="example_data",
        device="cuda",
        discretize_actions=False,  # Discretize the expert actions
        make_video=True,  # Record the trajectories as sanity check
        save_path="output_video.mp4",
    )

    # Save the expert actions and observations
    torch.save(expert_actions, "expert_actions.pt")
    torch.save(expert_obs, "expert_obs.pt")
