import numpy as np
import mediapy as media
import wandb
import torch

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from gpudrive.utils.checkpoint import load_agent

def test_rollout(focus_agents=[0, 1], render=False, agent=None):
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
        dynamics_model="classic",  # "state", #"classic",
        smoothen_trajectory=True,
        smoothness_weight=0.0,
        collision_weight=-0.02,
        off_road_weight=-0.0,
        guidance_heading_weight=0.005,
        guidance_speed_weight=0.0005,
        
    )

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/wosac/validation_json_1",
        batch_size=1,
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
    next_obs = env.reset(mask=control_mask)

    sim_frames = []
    agent_obs_frames = {i: [] for i in focus_agents}
    cum_reward = np.zeros((env.num_worlds, env.max_cont_agents))
    
    expert_actions, _, _, _ = env.get_expert_actions()

    for time_step in range(env.init_steps, env.episode_len):
        print(f"Step: {env.step_in_world[0, 0, 0].item()}")

        # Step
        if agent is not None: # Rollout with pre-trained agent
            control_mask = env.cont_agent_mask
            action, _, _, _ = agent(next_obs)
            action_template = torch.zeros(
                (1, 32), dtype=torch.int64, device='cuda'
            )
            action_template[control_mask] = action.to('cuda')
            
            env.step_dynamics(action_template)
            
        else:
            env.step_dynamics(expert_actions[:, :, time_step, :])

        next_obs = env.get_obs(control_mask)
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
            
        sim_frames_arr = np.array(sim_frames)
        wandb.log(
            {
                "sim/render": wandb.Video(
                    np.moveaxis(sim_frames_arr, -1, 1),
                    fps=5,
                    format="mp4",
                )
            }
        )

    env.close()
    run.finish()


if __name__ == "__main__":
    
    
    # load policy    
    CPT_PATH = (
        "checkpoints/model_guidance_logs__S_1__05_12_18_44_58_857_000100.pt"
    )
    agent = load_agent(path_to_cpt=CPT_PATH).to("cuda")

    
    
    test_rollout(focus_agents=[0, 1, 2, 3, 4], render=True, agent=agent)
