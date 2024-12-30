import torch
import numpy as np
import pandas as pd
import yaml
from box import Box
import dataclasses

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.dataset import SceneDataLoader

from integrations.rl.puffer.utils import Policy

def load_policy(model_path, device):
    """Load a policy from a given path."""

    saved_cpt = torch.load(model_path, map_location=device)

    # Build the model
    policy = ... # TODO

    # Load the model parameters
    policy.load_state_dict(saved_cpt["parameters"])

    return policy.eval()

def rollout(env, policy):
    """Rollout policy in the environment."""

    goal_achieved_list = []
    collided_list = []
    count_steps_to_goal = []
    running_dead_agent_ids = []

    # Reset
    next_obs = env.reset()

    # Initialize masks
    init_av_agent_mask = env.av_agent_mask.clone()
    dead_agent_mask = torch.zeros(env.num_envs, dtype=torch.bool).to(
        env.device
    )

    for time_step in range(env.config.episode_len):
        # Select actions for only active AV agents
        av_agent_actions, _, _, _ = policy(next_obs)

        # Step environment with active agent actions
        env.step_dynamics(av_agent_actions)

        # Gather new observations, dones, and infos for active agents
        dones = (
            env.sim.done_tensor().to_torch().squeeze(dim=2)[init_av_agent_mask]
        )
        infos = (
            env.sim.info_tensor().to_torch().squeeze(dim=2)[init_av_agent_mask]
        )

        current_dead_agent_ids = torch.where(dones == 1)[0].tolist()
        for agent_id in current_dead_agent_ids:
            # TODO: this is v inefficient
            if agent_id not in running_dead_agent_ids:
                running_dead_agent_ids.append(agent_id)
                # Update statistics
                goal_achieved = infos[agent_id, 3].item()
                goal_achieved_list.append(goal_achieved)
                collided_list.append(
                    infos[agent_id, 1].item() + infos[agent_id, 2].item()
                )
                if goal_achieved:
                    count_steps_to_goal.append(time_step)

        # Update dead agent mask
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        env.av_agent_mask[dead_agent_mask, :] = False

        # Get next obs for active agents only
        next_obs = env.get_obs()

        # End episode if all agents are dead
        if dead_agent_mask.all():
            break

    # Aggregate scores
    goal_achieved = sum(goal_achieved_list) / env.num_envs
    collision = sum(collided_list) / env.num_envs
    mean_steps_to_goal = np.array(count_steps_to_goal).mean()

    return {
        "goal_achieved": goal_achieved,
        "collision": collision,
        "mean_steps_to_goal": mean_steps_to_goal,
    }


def evaluate_policy(env, policy):
    """Evaluate policy in the environment."""

    res_dict = {}
    
    # For all batches of data
    # Evaluate the policy
    #res_dict[f"trial_{trial_i}"] = rollout_without_memory(env, policy)
        
    return pd.DataFrame(res_dict)

def load_config(cfg: str) -> Box:
    """Load configurations as a Box object.
    Args:
        cfg (str): Name of config file.

    Returns:
        Box: Box representation of configurations.
    """
    with open(f"{cfg}.yaml", "r") as stream:
        config = Box(yaml.safe_load(stream))
    return config

def make_env(config):
    """Make the environment with the given config."""
    
    scene_config = SceneConfig(
        path=config.data_dir,
        num_scenes=config.num_worlds,
        discipline=SelectionDiscipline.K_UNIQUE_N,
        k_unique_scenes=config.num_worlds,
        seed=config.sampling_seed,
    )

    # Override any default environment settings
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        norm_obs=config.normalize_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        dist_to_goal_threshold=config.dist_to_goal_threshold,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=config.use_lidar_obs,
        disable_classic_obs=True if config.use_lidar_obs else False,
        obs_radius=config.obs_radius,
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=config.max_controlled_agents,
        device=config.device,
    )
    
    return env

if __name__ == "__main__":
        
    # Make environment with the given config
    config = load_config("examples/experiments/eval/config/eval_base")
    env = make_env(config)
    
    # Make dataloaders
    train_loader = SceneDataLoader(
        root="data/processed/training",
        batch_size=100,
        dataset_size=1000,
        sample_with_replacement=False
    )
    
    test_loader = SceneDataLoader(
        root="data/processed/testing",
        batch_size=100,
        dataset_size=1000,
        sample_with_replacement=False
    )
    
    for batch, idx in enumerate(train_loader):
        print(len(batch))
    
    # # Load policies
    # policy = load_policy(
    #     model_path=config.model.baseline_model_path,
    #     device=config.setup.device,
    # )
    
    # # Evaluate
    # df_res = get_performance_over_trials(
    #     env,
    #     policy_baseline,
    # )

    # # Save results for analysis
    # dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # df_res.to_csv(f"{config.results.store_dir}/{dt_string}.csv")