"""Extract expert states and actions from Waymo Open Dataset."""
import torch
from pygpudrive.env.config import EnvConfig, RenderConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv


if __name__ == "__main__":

    # Constants
    TOTAL_STEPS = 10
    MAX_NUM_OBJECTS = 128
    NUM_WORLDS = 10

    render_config = RenderConfig()

    # Make environment with Waymax model
    env_config = EnvConfig(use_waymax_model=True, sample_method="pad_n")

    env = GPUDriveTorchEnv(
        config=env_config,
        num_worlds=NUM_WORLDS,
        max_cont_agents=MAX_NUM_OBJECTS,
        data_dir="example_data",
        device="cuda",
        render_config=render_config,
    )

    # Reset
    obs = env.reset()

    # Obtain expert actions for full trajectory in all worlds
    expert_actions = env.get_expert_actions()

    # TODO: Discretize expert actions

    # # Extract expert actions for the controlled (valid) agents
    # expert_actions = expert_actions[env.cont_agent_mask, :, :]

    # Storage
    obs_tensor = torch.zeros(
        (
            env.num_valid_controlled_agents_across_worlds,
            env.episode_len,
            env.observation_space.shape[0],
        ),
    )

    # Obtain the expert observations by stepping through the env
    # using the expert actions
    # TODO(dc) - rmv obs, actions from dead agents
    for time_step in range(env.episode_len):

        # Step using the expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])

        next_obs = env.get_obs()

        # Append (expert_actions, expert_obs) to the tensor
        obs_tensor[:, time_step, :] = next_obs[env.cont_agent_mask, :]

    # Save the expert actions and observations
    # ...
    # torch.save(expert_actions[env.cont_agent_mask, :, :], "expert_actions.pt")
    # torch.save(obs_tensor, "expert_obs.pt")
