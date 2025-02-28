import torch
import dataclasses
import mediapy
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.agents.core import merge_actions
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
import imageio
import numpy as np


def create_policy_masks(env, num_sim_agents=2):
    policy_mask = torch.zeros_like(env.cont_agent_mask, dtype=torch.int)
    agent_indices = env.cont_agent_mask.nonzero(as_tuple=True)

    for i, (world_idx, agent_idx) in enumerate(zip(*agent_indices)):
        policy_mask[world_idx, agent_idx] = (i % num_sim_agents) + 1

    policy_mask = {f'pi_{int(policy.item())}': (policy_mask == policy)
            for policy in policy_mask.unique() if policy.item() != 0}


    policy_world_mask = {
        world: {f'pi_{p+1}': policy_mask[f'pi_{p+1}'][world] for p in range(num_sim_agents)}
        for world in range(env.cont_agent_mask.shape[0])
    }
    return policy_world_mask
# Configs model has been trained with
config = load_config("../../examples/experimental/eval/config/reliable_agents_params")



max_agents = config.max_controlled_agents
num_envs = 2
device = "cpu" # cpu just because we're in a notebook
NUM_SIM_AGENTS = 2

sim_agent1 = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")
sim_agent2 = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")

# Some other info
card = ModelCard.load("daphne-cornelisse/policy_S10_000_02_27")



# Create data loader
train_loader = SceneDataLoader(
    root='../../gpudrive/data/processed/examples',
    batch_size=num_envs,
    dataset_size=100,
    sample_with_replacement=False,
)

# Set params
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
NUM_WORLDS = 2
# Make env
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=config.max_controlled_agents,
    device=device,
)

next_obs = env.reset()

control_mask = env.cont_agent_mask

action1, logprob1, entropy1, value1 = sim_agent1(
    next_obs[control_mask], deterministic=False
)

action2, logprob2, entropy2, value2 = sim_agent2(
    next_obs[control_mask], deterministic=False
)


if __name__ == "__main__":
    policy_masks= create_policy_masks(env,NUM_SIM_AGENTS)
    VIDEO_PATH = f"videos/"
    FPS = 23

    next_obs = env.reset()

    control_mask = env.cont_agent_mask

    print(next_obs.shape)

    frames = {f"env_{i}": [] for i in range(num_envs)}

    for time_step in range(env.episode_len):
        print(f"\rStep: {time_step}", end="", flush=True)


        action1, logprob1, entropy1, value1 = sim_agent1(
            next_obs[control_mask], deterministic=False
        )

        action2, logprob2, entropy2, value2 = sim_agent2(
            next_obs[control_mask], deterministic=False
        )


        actions_list = [
                        action1,
                        action2
                        ]

        action = torch.zeros(len(action1))

        action_idx = 0  # Starting index for filling the action tensor

        while action_idx < len(action):
            for i in range(NUM_SIM_AGENTS):
                if action_idx < len(action):
                    # Corrected indexing: select value from the i-th policy, using modulo for cycling
                    action[action_idx] = actions_list[i][action_idx % len(actions_list[i])]
                    action_idx += 1




        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        
        action_template[control_mask] = action.to(device).to(action_template.dtype)

        # Step
        env.step_dynamics(action_template)

        # Render    
        sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(NUM_WORLDS)),
                time_steps=[time_step]*NUM_WORLDS,
                zoom_radius=70,
                policy_masks=policy_masks
            )
        
        for i in range(num_envs):
            frames[f"env_{i}"].append(img_from_fig(sim_states[i])) 

        next_obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()
        
        if done.all():
            break

    env.close()
    for scene_name, frames_list in frames.items():
        frames_arr = np.array(frames_list)
        save_path = f"{VIDEO_PATH}{scene_name}.gif"
        print(f"Saving video at {save_path}")
        try:
            imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)
        except:
            import os
            os.mkdir(VIDEO_PATH)
            imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)