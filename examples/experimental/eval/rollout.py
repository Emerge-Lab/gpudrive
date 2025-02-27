from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig

import mediapy as media
import numpy as np
import torch

from eval_utils import load_config, load_policy, make_env


if __name__ == "__main__":

    eval_config = load_config("examples/experimental/eval/config/eval_config")
    render_config = RenderConfig()

    device = 'cuda'
    
    NUM_WORLDS = eval_config.num_worlds
    MAX_AGENTS = eval_config.max_controlled_agents
    

    # Create data loader
    train_loader = SceneDataLoader(
        root=eval_config.train_dir,
        batch_size=NUM_WORLDS,
        dataset_size=100,
        sample_with_replacement=True,
        shuffle=True,
    )

    # Make env
    env = make_env(eval_config, train_loader, render_3d=False)
    
    # Load policy
    policy = load_policy(
        path_to_cpt="examples/experimental/eval/models",
        model_name="model_PPO____S_800__02_24_15_17_45_245_002400",
        device=device,
        env=env,
    )

    control_mask = env.cont_agent_mask

    # Rollout
    next_obs = env.reset()

    sim_frames = []
    agent_obs_frames = []

    expert_actions, _, _, _ = env.get_expert_actions()

    env_idx = 0

    for t in range(10):
        print(f"Step: {t}")

        action, _, _, _ = policy(
            next_obs[control_mask], deterministic=False
        )

        # Insert actions into a template
        action_template = torch.zeros(
            (NUM_WORLDS, MAX_AGENTS), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)

        env.step_dynamics(action_template)
        
        highlight_agent = torch.where(control_mask[env_idx, :])[0][
            -1
        ].item()

        # Make video
        sim_states = env.vis.plot_simulator_state(
            env_indices=[env_idx],
            zoom_radius=100,
            time_steps=[t],
            #center_agent_indices=[highlight_agent],
        )

        agent_obs = env.vis.plot_agent_observation(
            env_idx=env_idx,
            agent_idx=highlight_agent,
            figsize=(10, 10),
        )
        
        sim_frames.append(img_from_fig(sim_states[0]))
        agent_obs_frames.append(img_from_fig(agent_obs))

        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()

        if done[0, highlight_agent].bool():
            break

    env.close()

    media.write_video(
        "sim_video.gif", np.array(sim_frames), fps=10, codec="gif"
    )
    media.write_video(
        "obs_video.gif", np.array(agent_obs_frames), fps=10, codec="gif"
    )
