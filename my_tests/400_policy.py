import torch
import imageio
import dataclasses
import mediapy
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config


config = load_config("/scratch/ra4924/gpudrive/my_tests/400_config")


max_agents = config.environment.max_controlled_agents
num_envs = 2
device = "cpu"


ckpt = torch.load("/scratch/ra4924/gpudrive/my_tests/model_PPO____S_75__05_21_11_20_13_859_004800.pt", 
            map_location=device,
            weights_only = False)


policy = NeuralNet(
    input_dim=ckpt["model_arch"]["input_dim"],
    action_dim=ckpt["action_dim"],
    hidden_dim=ckpt["model_arch"]["hidden_dim"],
    config=config.environment
).to(device)


policy.load_state_dict(ckpt["parameters"])
policy.eval()





train_loader = SceneDataLoader(
    root='data/processed/examples',
    batch_size=num_envs,
    dataset_size=100,
    sample_with_replacement=False,
)


env_config = dataclasses.replace(
    EnvConfig(),
    ego_state=config.environment.ego_state,
    road_map_obs=config.environment.road_map_obs,
    partner_obs=config.environment.partner_obs,
    reward_type=config.environment.reward_type,
    norm_obs=config.environment.norm_obs,
    dynamics_model=config.environment.dynamics_model,
    collision_behavior=config.environment.collision_behavior,
    dist_to_goal_threshold=config.environment.dist_to_goal_threshold,
    polyline_reduction_threshold=config.environment.polyline_reduction_threshold,
    remove_non_vehicles=config.environment.remove_non_vehicles,
    lidar_obs=config.environment.lidar_obs,
    disable_classic_obs=config.environment.lidar_obs,
    obs_radius=config.environment.obs_radius,
    # steer_actions = torch.round(
    #     torch.linspace(-torch.pi, torch.pi, config.environment.action_space_steer_disc), decimals=3  
    # ),
    # accel_actions = torch.round(
    #     torch.linspace(-4.0, 4.0, config.environment.action_space_accel_disc), decimals=3
    # ),
)


env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=config.environment.max_controlled_agents,
    device=device,
)





next_obs = env.reset()

control_mask = env.cont_agent_mask

print(next_obs.shape)

frames = {f"env_{i}": [] for i in range(num_envs)}

j = 0

while True:

    for time_step in range(env.episode_len):
        print(f"\rStep: {time_step}","/", env.episode_len, end="", flush=True)

        action, _, _, _ = policy(
            next_obs[control_mask], deterministic=False
        )
        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)

        env.step_dynamics(action_template)
  
        sim_states = env.vis.plot_simulator_state(
            env_indices=list(range(num_envs)),
            time_steps=[time_step]*num_envs,
            zoom_radius=70,
        )
        
        for i in range(num_envs):
            frames[f"env_{i}"].append(img_from_fig(sim_states[i])) 

        next_obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()
    
    if done.all():
        if j == 0:
            break
        else:
            j += 1
            print("Resetting envs")
            env.reset()
            next_obs = env.get_obs()
            control_mask = env.cont_agent_mask
            continue




# total_steps = 2        # how many frames/steps you want
# frames = []

# obs = env.reset()
# for t in range(total_steps):
#     print(f"\rStep: {t}", "/", total_steps, end="", flush=True)

#     # 1) Render
#     sim_states = env.vis.plot_simulator_state(
#         env_indices=list(range(num_envs)),
#         time_steps=[t % env.episode_len]*num_envs,
#         zoom_radius=70,
#     )
#     for i in range(num_envs):
#         frames.append(img_from_fig(sim_states[i]))

#     # 2) Act
#     action, *_ = policy(env.get_obs()[control_mask], deterministic=False)
#     template = torch.zeros((num_envs, max_agents), dtype=torch.int64, device=device)
#     template[control_mask] = action
#     env.step_dynamics(template)

#     # 3) If this episode ended, reset automatically
#     done = env.get_dones().all()
#     if done:
#         env.reset()


env.close()





# mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

combined = [np.hstack((frames["env_0"][t], frames["env_1"][t]))
            for t in range(len(frames["env_0"]))]

arr = np.stack(combined)

video_path = "/scratch/ra4924/gpudrive/ppo_run.mp4"


# imageio.mimwrite(video_path, arr, fps=15)  # no codec needed here



print(f"starting gif")

gif_path = "/scratch/ra4924/gpudrive/my_tests/ppo_run.gif"
# arr is your (T, H, W, 3) NumPy array of frames
# duration is seconds per frame (1/15 for 15 fps)
imageio.mimsave(gif_path, arr, format='GIF', duration=1/15)

print(f"Saved GIF to {gif_path}")





# mediapy.write_video(video_path, arr, fps=15, codec='libx264')
print(f"Saved video to {video_path}")