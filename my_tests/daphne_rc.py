
import torch
import dataclasses
import mediapy
from gpudrive.networks.late_fusion import NeuralNet
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState

from PIL import Image

def rollout(
    env,
    policy,
    device,
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 1,
    return_agent_positions: bool = False,
    return_behavior_metrics: bool = False,
    set_agent_type: bool = False,
    agent_weights: torch.Tensor = None,
    center_on_ego: bool = False,
    zoom_radius: int = 100,
):
    """
    Perform a rollout of a policy in the environment.

    Args:
        env: The simulation environment.
        policy: The policy to be rolled out.
        device: The device to execute computations on (CPU/GPU).
        deterministic (bool): Whether to use deterministic policy actions.
        render_sim_state (bool): Whether to render the simulation state.
        render_every_n_steps (int): Render every N steps.
        zoom_radius (int): Radius for zoom in visualization.
        return_agent_positions (bool): Whether to return agent positions.
        return_behavior_metrics (bool): Whether to collect behavioral metrics.
        center_on_ego (bool): Whether to center visualization on ego vehicle.
        set_agent_type (bool): Whether to set agent type during reset.
        agent_weights (torch.Tensor): Agent weights tensor for condition_mode="fixed".

    Returns:
        tuple: Averages for goal achieved, collisions, off-road occurrences,
               controlled agents count, simulation state frames, and (if requested) behavioral metrics.
    """
    # Initialize storage
    sim_state_frames = {env_id: [] for env_id in range(env.num_worlds)}
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len
    agent_positions = torch.zeros(
        (env.num_worlds, env.max_agent_count, episode_len, 2)
    )

    if return_behavior_metrics:
        entropy_values = torch.zeros(
            (num_worlds, max_agent_count, episode_len), device=device
        )
        logprob_values = torch.zeros(
            (num_worlds, max_agent_count, episode_len), device=device
        )
        # Assumes the action space is 3-dimensional (dynamics_model = "classic")
        action_history = torch.zeros(
            (num_worlds, max_agent_count, episode_len, 3), device=device
        )
    else:
        entropy_values, logprob_values, action_history = None, None, None

    # Reset episode
    if set_agent_type and agent_weights is not None:
        # Pass agent_weights to reset when set_agent_type is True
        next_obs = env.reset(condition_mode="fixed", agent_type=agent_weights)
    else:
        next_obs = env.reset()

    # Storage
    goal_achieved = torch.zeros((num_worlds, max_agent_count), device=device)
    collided = torch.zeros((num_worlds, max_agent_count), device=device)
    off_road = torch.zeros((num_worlds, max_agent_count), device=device)
    active_worlds = np.arange(num_worlds).tolist()
    episode_lengths = torch.zeros(num_worlds)

    control_mask = env.cont_agent_mask
    live_agent_mask = control_mask.clone()

    for time_step in range(episode_len):

        time_mask = torch.zeros(episode_len, dtype=torch.bool, device=device)
        time_mask[time_step] = True

        # Get actions for active agents
        if live_agent_mask.any():
            action, logprob, entropy, value = policy(
                next_obs[live_agent_mask], deterministic=deterministic
            )

            # Insert actions into a template
            action_template = torch.zeros(
                (num_worlds, max_agent_count), dtype=torch.int64, device=device
            )
            action_template[live_agent_mask] = action.to(device)

            if return_behavior_metrics:
                mask = live_agent_mask.unsqueeze(2) & time_mask.unsqueeze(
                    0
                ).unsqueeze(0)
                entropy_values[mask] = entropy
                logprob_values[mask] = logprob

                action_values = env.action_keys_tensor[action]
                action_history[mask] = action_values

            # Step the environment
            env.step_dynamics(action_template)

            # Render
            if render_sim_state and len(active_worlds) > 0:
                has_live_agent = torch.where(
                    live_agent_mask[active_worlds, :].sum(axis=1) > 0
                )[0].tolist()

                if time_step % render_every_n_steps == 0:
                    if center_on_ego:
                        agent_indices = torch.argmax(
                            control_mask.to(torch.uint8), dim=1
                        ).tolist()
                    else:
                        agent_indices = None

                    sim_state_figures = env.vis.plot_simulator_state(
                        env_indices=has_live_agent,
                        time_steps=[time_step] * len(has_live_agent),
                        zoom_radius=zoom_radius,
                        center_agent_indices=agent_indices,
                    )
                    for idx, env_id in enumerate(has_live_agent):
                        sim_state_frames[env_id].append(
                            img_from_fig(sim_state_figures[idx])
                        )

        # Update observations, dones, and infos
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()

        off_road[live_agent_mask] += infos.off_road[live_agent_mask]
        collided[live_agent_mask] += infos.collided[live_agent_mask]
        goal_achieved[live_agent_mask] += infos.goal_achieved[live_agent_mask]

        # Update live agent mask
        live_agent_mask[dones] = False

        # Process completed worlds
        num_dones_per_world = (dones & control_mask).sum(dim=1)
        total_controlled_agents = control_mask.sum(dim=1)
        done_worlds = (num_dones_per_world == total_controlled_agents).nonzero(
            as_tuple=True
        )[0]

        for world in done_worlds:
            if world in active_worlds:
                active_worlds.remove(world)
                episode_lengths[world] = time_step

        if return_agent_positions:
            global_agent_states = GlobalEgoState.from_tensor(
                env.sim.absolute_self_observation_tensor()
            )
            agent_positions[:, :, time_step, 0] = global_agent_states.pos_x
            agent_positions[:, :, time_step, 1] = global_agent_states.pos_y

        if not active_worlds:  # Exit early if all worlds are done
            break

    # Aggregate metrics to obtain averages across scenes
    controlled_per_scene = control_mask.sum(dim=1).float()

    # Counts
    goal_achieved_count = (goal_achieved > 0).float().sum(axis=1)
    collided_count = (collided > 0).float().sum(axis=1)
    off_road_count = (off_road > 0).float().sum(axis=1)
    not_goal_nor_crash_count = (
        torch.logical_and(
            goal_achieved == 0,  # Didn't reach the goal
            torch.logical_and(
                collided == 0,  # Didn't collide
                torch.logical_and(
                    off_road == 0,  # Didn't go off-road
                    control_mask,  # Only count controlled agents
                ),
            ),
        )
        .float()
        .sum(dim=1)
    )

    # Fractions per scene
    frac_goal_achieved = goal_achieved_count / controlled_per_scene
    frac_collided = collided_count / controlled_per_scene
    frac_off_road = off_road_count / controlled_per_scene
    frac_not_goal_nor_crash_per_scene = (
        not_goal_nor_crash_count / controlled_per_scene
    )

    # Prepare return values
    base_metrics = (
        goal_achieved_count,
        frac_goal_achieved,
        collided_count,
        frac_collided,
        off_road_count,
        frac_off_road,
        not_goal_nor_crash_count,
        frac_not_goal_nor_crash_per_scene,
        controlled_per_scene,
        sim_state_frames,
        agent_positions,
        episode_lengths,
    )

    # Return behavioral metrics if requested
    if return_behavior_metrics:
        behavior_metrics = {
            "entropy": entropy_values,
            "logprob": logprob_values,
            "actions": action_history,
        }
        return base_metrics + (behavior_metrics,)

    return base_metrics

def load_policy(path_to_cpt, model_name, device, env_config=None):
    """Load a policy from a given path."""

    # Load a trained model
    saved_cpt = torch.load(
        f=f"{path_to_cpt}/{model_name}.pt",
        map_location=device,
        weights_only=False,
    )

    print(f"Load model from {path_to_cpt}/{model_name}.pt")

    # Create policy architecture from saved checkpoint
    policy = NeuralNet(
        input_dim=saved_cpt["model_arch"]["input_dim"],
        action_dim=saved_cpt["action_dim"],
        hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        config=env_config,
    ).to(device)

    # Load the model parameters
    policy.load_state_dict(saved_cpt["parameters"])

    return policy.eval()


# Configs model has been trained with
config = load_config("my_tests/conditioned_agent")
num_envs = 4
device = "cpu"

config.environment.condition_mode = "fixed"
config.environment.agent_type = torch.Tensor([-0.2, 1.0, -0.2])


agent = load_policy(
    model_name="rew_conditioned_0321",
    path_to_cpt="my_tests",
    env_config=config.environment,
    device=device
)


# Create data loader
train_loader = SceneDataLoader(
    root='/scratch/ra4924/gpudrive/data/processed/examples',
    batch_size=num_envs,
    dataset_size=100,
    sample_with_replacement=False,
)

# Set params
config = config.environment
env_config = dataclasses.replace(
    EnvConfig(),
    norm_obs=config.norm_obs,
    dynamics_model=config.dynamics_model,
    collision_behavior=config.collision_behavior,
    dist_to_goal_threshold=config.dist_to_goal_threshold,
    polyline_reduction_threshold=config.polyline_reduction_threshold,
    remove_non_vehicles=config.remove_non_vehicles,
    lidar_obs=config.lidar_obs,
    disable_classic_obs=config.lidar_obs,
    obs_radius=config.obs_radius,
    # steer_actions = torch.round(
    #     torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), decimals=3
    # ),
    # accel_actions = torch.round(
    #     torch.linspace(-4.0, 4.0, config.action_space_accel_disc), decimals=3
    # ),
    action_space_steer_disc = config.action_space_steer_disc,
    action_space_accel_disc = config.action_space_accel_disc,
    reward_type=config.reward_type,
    condition_mode=config.condition_mode,
    agent_type=config.agent_type
)

# Make env
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=64, #config.max_controlled_agents,
    device=device,
)



def run_multiple_rollouts(
    env, 
    agent, 
    num_rollouts=2, 
    device='cpu',
    sample_collision_weights=True,
    sample_goal_weights=False,
    sample_offroad_weights=False,
    agent_type=None,
):
    """
    Run multiple rollouts with different collision weights and store trajectories.
    Stores agent positions as a tensor of shape [num_envs, num_rollouts, agents, steps, 2].
    
    Args:
        env: The environment (can be batched with multiple environments)
        agent: The policy
        num_rollouts: Number of rollouts to perform
        device: Device to run on
        
    Returns:
        all_trajectories: Dictionary containing trajectories and weights
    """
    
    all_agent_positions = []
    collision_weights = []
    goal_weights = []
    offroad_weights = []
    all_goal_achieved = []
    all_collided = []
    all_off_road = []
    all_episode_lengths = []
    
    print(f"Running {num_rollouts} rollouts, sampling weights: collision={sample_collision_weights}, goal={sample_goal_weights}, offroad={sample_offroad_weights}\n")
    
    for i in tqdm(range(num_rollouts), desc="Processing rollouts", unit="rollout"):
        
        # Sample weights
        if agent_type is not None:
            agent_weights = agent_type
            collision_weight = agent_weights[0].item()
            goal_weight = agent_weights[1].item()
            off_road_weight = agent_weights[2].item()
        else:
            if sample_collision_weights:
                collision_weight = random.uniform(-3.0, 1.0)
            else:
                collision_weight = -3.0
            if sample_goal_weights:
                goal_weight = random.uniform(1.0, 3.0)
            else:
                goal_weight = 1.0
            if sample_offroad_weights:
                off_road_weight = random.uniform(-3.0, 1.0)
            else:
                off_road_weight = -3.0

            agent_weights = torch.Tensor([collision_weight, goal_weight, off_road_weight])
        
        # Run rollout with these weights
        (
            goal_achieved_count,
            frac_goal_achieved,
            collided_count,
            frac_collided,
            off_road_count,
            frac_off_road,
            not_goal_nor_crash_count,
            frac_not_goal_nor_crash_per_scene,
            controlled_agents_per_scene,
            sim_state_frames,
            agent_positions,
            episode_lengths
        ) = rollout(
            env=env,
            policy=agent,
            device=device,
            deterministic=False,
            return_agent_positions=True,
            set_agent_type=True,
            agent_weights=agent_weights,
        )
        
        # Store weights and positions
        collision_weights.append(collision_weight)
        goal_weights.append(goal_weight)
        offroad_weights.append(off_road_weight)
        all_agent_positions.append(agent_positions.clone().detach())
        
        # Store other metrics
        all_goal_achieved.append(goal_achieved_count)
        all_collided.append(collided_count)
        all_off_road.append(off_road_count)
        all_episode_lengths.append(episode_lengths)
    
    # Stack agent positions along a new dimension at position 1 (after num_envs)
    # From list of [num_envs, num_agents, time_steps, 2] to tensor of [num_envs, num_rollouts, num_agents, time_steps, 2]
    stacked_positions = torch.stack(all_agent_positions, dim=1)
    
    # Return organized data
    all_trajectories = {
        'collision_weights': torch.tensor(collision_weights),
        'goal_weights': torch.tensor(goal_weights),
        'offroad_weights': torch.tensor(offroad_weights),
        'agent_positions': stacked_positions,  # Shape: [num_envs, num_rollouts, num_agents, time_steps, 2]
        'goal_achieved': all_goal_achieved,
        'collided': all_collided,
        'off_road': all_off_road,
        'episode_lengths': all_episode_lengths
    }
    
    return all_trajectories



# Define different agent types to compare
agent_configs = { # Collision, Goal, Off-road
    'Nominal': torch.tensor([-0.75, 1.0, -0.75], device=device),
    'Aggressive': torch.tensor([0.0, 2.0, 0.0], device=device),
    'Risk-averse': torch.tensor([-2.0, 0.5, -2.0], device=device),
}




trajs = run_multiple_rollouts(
    env=env,
    agent=agent,
    num_rollouts=2,
    device='cpu',
    sample_collision_weights=True,
    sample_goal_weights=False,
    sample_offroad_weights=False,
    #agent_type=agent_configs['Risk-averse'],
)




env.vis.figsize = (8, 8)

# Plot simulator state with the stacked trajectories
_ = env.reset(agent_type=torch.Tensor([-0.2, 1.0, -0.2]))
img = env.vis.plot_simulator_state(
    env_indices=[1], 
    agent_positions=trajs['agent_positions'],  # Pass stacked trajectories directly
    zoom_radius=70,
    multiple_rollouts=True,
    line_alpha=0.5,          
    line_width=1.0,     
    weights=trajs['collision_weights'],     
    colorbar=True, 
)[0]

Image.fromarray(img_from_fig(img))


print("yalah activi dakshi li b9a  a7madiiiii22")
fig = img  
filename = "effect_of_rew_cond.png"
fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Figure saved as {filename}")



agent_configs = {
    'Nominal': torch.tensor([-0.5, 0.1, -0.5], device=device),
}

print("Collecting rollout data...")
collected_data = {}

agent_weights = torch.Tensor([-0.2, 1.0, -0.2])


div_metrics = collect_rollout_data(
    env, agent, agent_weights, device, 2
)

df = store_data_to_dataframe(list(agent_configs.keys()), collected_data)



div_metrics.keys()


div_metrics['entropy_values'].shape

