############################ SET UP LIBRARIES ############################

import os
from pathlib import Path

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

import torch
import dataclasses
import mediapy
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config


'''

############################ SET UP CONFIG AND ENVIRONMENT ############################

# Configs model has been trained with
config = load_config("examples/experimental/config/reliable_agents_params")
config.max_controlled_agents = 1

# print(config)

max_agents = config.max_controlled_agents
num_envs = 1
device = "cpu" # cpu just because we're in a notebook
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Load pre-trained agent via Hugging Face hub

sim_agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")

## Agent has an action dimension of 91: 13 steering wheel angle discretizations x 9 acceleration discretizations
sim_agent.action_dim

## Size of flattened observation vector
sim_agent.obs_dim



# Make Environment

## Create data loader
train_loader = SceneDataLoader(
    root='data/processed/examples',
    batch_size=num_envs,
    dataset_size=100,
    sample_with_replacement=False,
)

## Set params
env_config = dataclasses.replace(
    EnvConfig(),
    ego_state=config.ego_state,
    road_map_obs=config.road_map_obs,
    partner_obs=config.partner_obs,
    reward_type=config.reward_type,
    norm_obs=config.norm_obs,
    dynamics_model=config.dynamics_model,
    collision_behavior=config.collision_behavior,
    dist_to_goal_threshold=config.dist_to_goal_threshold,
    polyline_reduction_threshold=config.polyline_reduction_threshold,
    remove_non_vehicles=config.remove_non_vehicles,
    lidar_obs=config.lidar_obs,
    disable_classic_obs=config.lidar_obs,
    obs_radius=config.obs_radius,
    steer_actions = torch.round(
        torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), decimals=3  
    ),
    accel_actions = torch.round(
        torch.linspace(-4.0, 4.0, config.action_space_accel_disc), decimals=3
    ),
)

## Make env
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=config.max_controlled_agents,
    device=device,
)




############################ ROLL OUT POLICY ############################

construal_size = 16
for const_num in range(int(max_agents/construal_size)):
    # Repeat rollout for each construal size

    ## Defnie observation mask for construal
    obs_mask = [False]*max_agents
    obs_start_indx = int(const_num*construal_size)
    obs_end_indx = int((const_num+1)*construal_size)
    obs_mask[obs_start_indx:obs_end_indx] = [True]*construal_size

    next_obs = env.reset()
    control_mask = env.cont_agent_mask
    print("Observation shape: ", next_obs.shape)
    frames = {f"env_{i}-constr_{const_num}": [] for i in range(num_envs)}
    
    for time_step in range(env.episode_len):
        ## Roll out policy for a specific construal
        print(f"\rStep: {time_step}", end="", flush=True)

        ### Predict actions
        action, _, _, _ = sim_agent(
            next_obs[control_mask], deterministic=False
        )
        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)

        ### Step
        env.step_dynamics(action_template)

        ### Render    
        sim_states = env.vis.plot_simulator_state(
            env_indices=list(range(num_envs)),
            time_steps=[time_step]*num_envs,
            zoom_radius=70,
        )
        
        for i in range(num_envs):
            frames[f"env_{i}"].append(img_from_fig(sim_states[i])) 

        next_obs = env.get_obs(obs_mask)
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()
        
        if done.all():
            break

    ## Save animations
    mediapy.set_show_save_dir('./sim_vids')
    mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

env.close()

'''