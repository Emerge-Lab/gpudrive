"""Extract expert states and actions from Waymo Open Dataset."""
import torch
from pygpudrive.env.config import EnvConfig, RenderConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv


def generate_state_action_pairs(
    env_config,
    render_config,
    max_num_objects,
    num_worlds,
    data_dir,
    device,
    discretize_actions=False,
    use_heading=False,
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
        use_heading (bool): Whether to use heading information in the expert actions.

    Returns:
        expert_actions: Expert actions for the controlled agents.
        obs_tensor: Expert observations for the controlled agents.
    """

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
        # TODO(dc): Implement discretization
        expert_actions = env.discretize_actions(expert_actions)

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

        if (dead_agent_mask == True).all():
            break

    # Convert lists to tensors
    # Flatten the lists
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
    )

    # Save the expert actions and observations
    torch.save(expert_actions, "expert_actions.pt")
    torch.save(expert_obs, "expert_obs.pt")
