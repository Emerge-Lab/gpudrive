import numpy as np
import mediapy as media
import wandb

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig


def test_rollout(focus_agents=[0, 1], render=False):
    run = wandb.init(project="humanlike_tests", group="rollout_tests")

    env_config = EnvConfig(
        guidance=True,
        guidance_mode="log_replay",  # Options: "log_replay", "vbd_amortized"
        add_reference_pos_xy=True,
        add_reference_speed=True,
        add_reference_heading=True,
        add_previous_action=True,
        reward_type="guided_autonomy",
        init_mode="wosac_train",
        dynamics_model="delta_local",  # "state", #"classic",
        smoothen_trajectory=True,
        smoothness_weight=0.001,
        collision_weight=-0.01,
        off_road_weight=-0.01,
        guidance_heading_weight=0.01,
        guidance_speed_weight=0.01,
    )

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/wosac/validation_json_100",
        batch_size=100,
        dataset_size=100,
        sample_with_replacement=False,
        shuffle=False,
        file_prefix="",
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=32,  # Number of agents to control
        device="cuda",
    )

    control_mask = env.cont_agent_mask

    # Rollout
    obs = env.reset(mask=control_mask)

    sim_frames = []
    agent_obs_frames = {i: [] for i in focus_agents}
    cum_reward = np.zeros((env.num_worlds, env.max_cont_agents))

    expert_actions, _, _, _ = env.get_expert_actions()

    for time_step in range(env.init_steps, env.episode_len):
        print(f"Step: {env.step_in_world[0, 0, 0].item()}")

        # Step
        expert_actions, _, _, _ = env.get_expert_actions()

        # Sample random actions
        # expert_actions = np.random.uniform(
        #     low=-env_config.max_accel_value,

        env.step_dynamics(expert_actions[:, :, time_step, :])

        obs = env.get_obs(control_mask)
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()

        # for agent_idx in focus_agents:
        #     cum_reward[agent_idx] += reward[0, agent_idx].item()

        cum_reward += reward.cpu().numpy()

        if render:
            if time_step % 5 == 0 or time_step > env.episode_len - 3:
                sim_states, agent_obs = env.render(
                    focus_agent_idx=focus_agents
                )
                sim_frames.append(img_from_fig(sim_states[0]))
                for i in focus_agents:
                    agent_obs_frames[i].append(img_from_fig(agent_obs[i]))

        # Log reward magnitudes
        for agent_idx in focus_agents:
            agent_key = f"agent_{agent_idx}"
            wandb.log(
                {
                    f"{agent_key}/R_combined": reward[0, agent_idx].item(),
                    f"{agent_key}/R_jerk": env.smoothness_penalty[
                        0, agent_idx
                    ].item(),
                    f"{agent_key}/R_collision": env.base_rewards[
                        0, agent_idx
                    ].item(),
                    f"{agent_key}/R_speed_heading": env.speed_heading_reward[
                        0, agent_idx
                    ].item(),
                    f"{agent_key}/R_route": env.route_reward[
                        0, agent_idx
                    ].item(),
                    f"{agent_key}/R_cumulative": cum_reward[
                        0, agent_idx
                    ].item(),
                },
                step=env.step_in_world[0, 0, 0].item(),
            )
            ""
    avg_cum_reward = cum_reward[env.cont_agent_mask.cpu().numpy()].mean()

    print(
        f"Avg cumulative rewards N = {env.cont_agent_mask.sum()}: {avg_cum_reward}"
    )
    if render:
        for agent_idx in focus_agents:
            agent_key = f"agent_{agent_idx}"
            agent_obs_arr = np.array(agent_obs_frames[agent_idx])
            wandb.log(
                {
                    f"{agent_key}/render": wandb.Video(
                        np.moveaxis(agent_obs_arr, -1, 1),
                        fps=5,
                        format="mp4",
                    )
                }
            )

    env.close()
    run.finish()


if __name__ == "__main__":
    test_rollout(focus_agents=[0, 1, 2, 3, 4], render=True)
