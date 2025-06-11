import torch
import pandas as pd
from tqdm import tqdm
import yaml
from box import Box
import numpy as np
import dataclasses

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState

from gpudrive.networks.late_fusion import NeuralNet

import logging
import torch

logging.basicConfig(level=logging.INFO)

import pdb


class RandomPolicy:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n

    def __call__(self, obs, deterministic=False):
        """Generate random actions."""
        # Uniformly sample integers from the action space for each observation
        batch_size = obs.shape[0]
        random_action = torch.randint(
            0, self.action_space_n, (batch_size,), dtype=torch.int64
        )
        return random_action, None, None, None


def load_policy(path_to_cpt, model_name, device, env=None):
    """Load a policy from a given path."""

    # Load the saved checkpoint
    if model_name == "random_baseline":
        return RandomPolicy(env.action_space.n)

    else:  # Load a trained model
        saved_cpt = torch.load(
            f=f"{path_to_cpt}/{model_name}.pt",
            map_location=device,
            weights_only=False,
        )

        logging.info(f"Load model from {path_to_cpt}/{model_name}.pt")

        # Create policy architecture from saved checkpoint
        policy = NeuralNet(
            input_dim=saved_cpt["model_arch"]["input_dim"],
            action_dim=saved_cpt["action_dim"],
            hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        ).to(device)

        # Load the model parameters
        policy.load_state_dict(saved_cpt["parameters"])

        logging.info("Load model parameters")

        return policy.eval()

def rollout(
    env,
    policy,
    device,
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 1,
    zoom_radius: int = 100,
    return_agent_positions: bool = False,
    center_on_ego: bool = False,
):
    """
    Perform a rollout of a policy in the environment.

    Args:
        env: The simulation environment.
        policy: The policy to be rolled out.
        device: The device to execute computations on (CPU/GPU).
        deterministic (bool): Whether to use deterministic policy actions.
        render_sim_state (bool): Whether to render the simulation state.

    Returns:
        tuple: Averages for goal achieved, collisions, off-road occurrences,
               controlled agents count, and simulation state frames.
    """
    # Initialize storage
    sim_state_frames = {env_id: [] for env_id in range(env.num_worlds)}
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len
    agent_positions = torch.zeros((env.num_worlds, env.max_agent_count, episode_len, 2))
    
    # Reset episode
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
        
        print(f't: {time_step}')
        
        # Get actions for active agents
        if live_agent_mask.any():
            action, _, _, _ = policy(
                next_obs[live_agent_mask], deterministic=deterministic
            )

            # Insert actions into a template
            action_template = torch.zeros(
                (num_worlds, max_agent_count), dtype=torch.int64, device=device
            )
            action_template[live_agent_mask] = action.to(device)

            # Step the environment
            env.step_dynamics(action_template)

            # Render
            if render_sim_state and len(active_worlds) > 0:
                
                has_live_agent = torch.where(
                    live_agent_mask[active_worlds, :].sum(axis=1) > 0
                )[0].tolist()

                if time_step % render_every_n_steps == 0:
                    if center_on_ego:
                        agent_indices = torch.argmax(control_mask.to(torch.uint8), dim=1).tolist()
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
            global_agent_states = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor())
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
    not_goal_nor_crash_count = torch.logical_and(
        goal_achieved == 0,  # Didn't reach the goal
        torch.logical_and(
            collided == 0,  # Didn't collide
            torch.logical_and(
                off_road == 0,  # Didn't go off-road
                control_mask,  # Only count controlled agents
            ),
        ),
    ).float().sum(dim=1)
    
    # Fractions per scene
    frac_goal_achieved =  goal_achieved_count / controlled_per_scene
    frac_collided = collided_count / controlled_per_scene
    frac_off_road = off_road_count / controlled_per_scene
    frac_not_goal_nor_crash_per_scene = not_goal_nor_crash_count / controlled_per_scene

    return (
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


def make_env(config, train_loader, render_3d=False):
    """Make the environment with the given config."""

    # Override any default environment settings
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
        disable_classic_obs=True if config.lidar_obs else False,
        obs_radius=config.obs_radius,
        steer_actions = torch.round(
            torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), decimals=3  
        ),
        accel_actions = torch.round(
            torch.linspace(-4.0, 4.0, config.action_space_accel_disc), decimals=3
        ),
    )

    render_config =  RenderConfig()
    render_config.render_3d = render_3d

    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=config.max_controlled_agents,
        device=config.device,
        render_config=render_config
    )

    return env


def evaluate_policy(
    env,
    policy,
    data_loader,
    dataset_name,
    device="cuda",
    deterministic=False,
    render_sim_state=False,
):
    """Evaluate policy in the environment."""

    res_dict = {
        "scene": [],
        "goal_achieved_count": [],
        "goal_achieved_frac": [],
        "collided_count": [],
        "collided_frac": [],
        "off_road_count": [],
        "off_road_frac": [],
        "other_count": [],
        "other_frac": [],
        "controlled_agents_in_scene": [],
        "episode_lengths": [],
    }

    for batch in tqdm(
        data_loader,
        desc=f"Processing {dataset_name} batches",
        total=len(data_loader),
        colour="blue",
    ):

        # Update simulator with the new batch of data
        env.swap_data_batch(batch)

        # Rollout policy in the environments
        (
            goal_achieved_count,
            goal_achieved_frac,
            collided_count,
            collided_frac,
            off_road_count,
            off_road_frac,
            other_count,
            other_frac,
            controlled_agents_in_scene,
            sim_state_frames,
            agent_positions,
            episode_lengths,
        ) = rollout(
            env=env,
            policy=policy,
            device=device,
            deterministic=deterministic,
            render_sim_state=render_sim_state,
        )

        # Get names from env
        scenario_to_worlds_dict = env.get_env_filenames()

        res_dict["scene"].extend(scenario_to_worlds_dict.values())
        res_dict["goal_achieved_count"].extend(goal_achieved_count.cpu().numpy())
        res_dict["goal_achieved_frac"].extend(goal_achieved_frac.cpu().numpy())
        
        res_dict["collided_count"].extend(collided_count.cpu().numpy())
        res_dict["collided_frac"].extend(collided_frac.cpu().numpy())
        
        res_dict["off_road_count"].extend(off_road_count.cpu().numpy())
        res_dict["off_road_frac"].extend(off_road_frac.cpu().numpy())
        
        res_dict["other_count"].extend(other_count.cpu().numpy())
        res_dict["other_frac"].extend(other_frac.cpu().numpy())
        res_dict["controlled_agents_in_scene"].extend(
            controlled_agents_in_scene.cpu().numpy()
        )
        res_dict["episode_lengths"].extend(episode_lengths.cpu().numpy())

    # Convert to pandas dataframe
    df_res = pd.DataFrame(res_dict)
    df_res["dataset"] = dataset_name

    return df_res



def multi_policy_rollout(
    env,
    policies, 
    device,
    trials: int = 1,
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 1,
    zoom_radius: int = 100,
    return_agent_positions: bool = False,
    center_on_ego: bool = False,
    collision_weight: float = -0.75,
    off_road_weight: float = 0-0.75,
    goal_achieved_weight: float = 1.0
):
    """
    Perform a rollout of multiple policies in the environment.

    Args:
        env: The simulation environment.
        policies (dict): Dictionary of policies {policy_name: (policy_function,mask)}.
        device: The device to execute computations on (CPU/GPU).
        policy_masks (dict): Dictionary of policy masks {policy_name: mask_tensor}.
        deterministic (bool): Whether to use deterministic policy actions.
        return_agent_positions (bool): Whether to return agent positions.

    Returns:
        policy_metrics: Dictionary of metrics corresponding to policies {policy_name: metrics(dict)}
            metrics: {
                'goal_achieved', 'collided', 'off_road', 'off_road_count', 'collided_count', 'goal_achieved_count', 'frac_off_road', 'frac_collided', 'frac_goal_achieved'
                }

    """
    def compute_metrics(policy_metrics,policies):
        for policy_name, policy_data in policies.items():
            for trial in range(trials):
                
                controlled_per_scene = policy_data['mask'].sum(dim=1)
                


                policy_metrics[policy_name][trial]['off_road_count'] = (policy_metrics[policy_name][trial]["off_road"] > 0).float().sum(axis=1)
                policy_metrics[policy_name][trial]['collided_count'] = (policy_metrics[policy_name][trial]["collided"] > 0).float().sum(axis=1)
                policy_metrics[policy_name][trial]['goal_achieved_count'] = (policy_metrics[policy_name][trial]['goal_achieved'] > 0).float().sum(axis=1)
                policy_metrics[policy_name][trial]['reward_count'] = (policy_metrics[policy_name][trial]['reward'] > 0).float().sum(axis=1)

                policy_metrics[policy_name][trial]['off_road_per_world'] = policy_metrics[policy_name][trial]['off_road_count'] / controlled_per_scene
                policy_metrics[policy_name][trial]['collided_per_world'] = policy_metrics[policy_name][trial]['collided_count'] / controlled_per_scene 
                policy_metrics[policy_name][trial]['goal_achieved_per_world'] = policy_metrics[policy_name][trial]['goal_achieved_count'] / controlled_per_scene
                policy_metrics[policy_name][trial]['reward_per_world'] = policy_metrics[policy_name][trial]['reward_count'] / controlled_per_scene

                policy_metrics[policy_name][trial]['frac_off_road'] = torch.mean(policy_metrics[policy_name][trial]['off_road_per_world'])
                policy_metrics[policy_name][trial]['frac_collided'] = torch.mean(policy_metrics[policy_name][trial]['collided_per_world'])
                policy_metrics[policy_name][trial]['frac_goal_achieved'] = torch.mean(policy_metrics[policy_name][trial]['goal_achieved_per_world'])
                policy_metrics[policy_name][trial]['frac_reward'] = torch.mean(policy_metrics[policy_name][trial]['reward_per_world'])
                # Add standard deviations
                policy_metrics[policy_name][trial]['frac_off_road_std'] = torch.std(policy_metrics[policy_name][trial]['off_road_per_world'])
                policy_metrics[policy_name][trial]['frac_collided_std'] = torch.std(policy_metrics[policy_name][trial]['collided_per_world'])
                policy_metrics[policy_name][trial]['frac_goal_achieved_std'] = torch.std(policy_metrics[policy_name][trial]['goal_achieved_per_world'])
                policy_metrics[policy_name][trial]['frac_reward_std'] = torch.std(policy_metrics[policy_name][trial]['reward_per_world'])
            

        return policy_metrics

    def initialise_history(k_trials,episode_len,log_history_step,partner_obs_shape,device): #self.k_trials *(int(self.num_steps/self.log_history_step)+1) * self.max_observable_agents , constants.PARTNER_FEAT_DIM
        # history_dict = torch.full(
        #     (k_trials*int((episode_len/log_history_step)+1), partner_obs_shape), 0.0, device=device)
        history_dict = {f"trial_{k}": torch.full(
                        (int(episode_len/log_history_step)+1, partner_obs_shape), 
                        0.0, 
                        device=device
                    )
                    for k in range(k_trials)
                }
        return history_dict


    def update_history(history_dict,partner_obs,log_history_step,current_step, trial):

        if current_step % log_history_step ==0:

            step_index = current_step // log_history_step
            history_dict[f"trial_{trial}"][step_index] = partner_obs

        
        return history_dict
                
       

    def get_history_batch(history_dict,k_trials,observations,device):
        history_batch =            torch.stack([
                history_dict[f"trial_{k}"] 
                for k in range(k_trials)
            ])

        flattened_history_batch = history_batch.flatten(start_dim=0) 
        combination = torch.cat((observations,flattened_history_batch))
        return combination.to(device)
    
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = 91
    sim_state_frames = {env_id: [] for env_id in range(num_worlds)}
    agent_positions = torch.zeros((num_worlds, max_agent_count, episode_len, 2))

    next_obs = env.reset()
    policy_metrics = {
        policy_name: {
            trial: {
                "goal_achieved": torch.zeros((num_worlds, max_agent_count), device=device),
                "collided": torch.zeros((num_worlds, max_agent_count), device=device),
                "off_road": torch.zeros((num_worlds, max_agent_count), device=device),
                "reward": torch.zeros((num_worlds, max_agent_count), device=device),
            }
            for trial in range(trials)  
        }
        for policy_name in policies
    }
    episode_lengths = torch.zeros(num_worlds)
    
    active_worlds = list(range(num_worlds))
    control_mask = env.cont_agent_mask
    live_agent_mask = control_mask.clone()
    all_partner_observations = env._get_partner_obs()
    partner_obs_dim = all_partner_observations.shape[2]

    for policy,policy_data in policies.items():
        if 'history' in policy_data:
            controlled_indices = torch.nonzero(policy_data['mask']).squeeze()
            controlled_keys = [tuple(indice.tolist()) for indice in controlled_indices]
            policy_data['history_dicts'] ={}
            for keys in controlled_keys:
                policy_data['history_dicts'][keys] =initialise_history(trials,
                                                                       episode_len,
                                                                policy_data['history']['log_history'],
                                                                partner_obs_dim,
                                                                device)
            policy_data['env_to_step_in_trial_dict'] = {
            f"world_{i}": 0 for i in range(num_worlds)
            }


    for trial in range(trials):
        episode_lengths = torch.zeros(num_worlds)
        next_obs = env.reset()
            
        active_worlds = list(range(num_worlds))
        control_mask = env.cont_agent_mask
        live_agent_mask = control_mask.clone()

        for policy,policy_data in policies.items():
            policy_data['env_to_step_in_trial_dict'] = {
            f"world_{i}": 0 for i in range(num_worlds)
            }


        
        for time_step in range(episode_len):
            print(f't: {time_step}')

            policy_live_masks = {name: policy_data['mask'] & live_agent_mask for name, policy_data in policies.items()}

            all_partner_observations = env._get_partner_obs()
            actions = {}
            for policy_name, policy_data in policies.items():
                policy_fn = policy_data['func']
                live_mask = policy_live_masks[policy_name]
                if live_mask.any():
                    agent_observations = next_obs[live_mask]
                    if 'history' in policy_data:
      
                        live_indices = torch.nonzero(live_mask).view(-1, live_mask.dim())
                        live_keys = [tuple(indice.tolist()) for indice in live_indices]
                        observations_list = []

                        for idx, key in enumerate(live_keys):
                            observation = agent_observations[idx]
                            agent_observation = get_history_batch(policy_data['history_dicts'][key],
                                                                trials,
                                                                observation,
                                                                device)

                            observations_list.append(agent_observation)

                        agent_observations = torch.vstack(observations_list)


                    agent_actions, _, _, _  = policy_fn(
                        agent_observations, deterministic=deterministic
                    )
                    #print(f"agent actions {agent_actions}")
                    actions[policy_name]= agent_actions
                    if 'history' in policy_data:
                        partner_obs = all_partner_observations[live_agent_mask]
                        for idx,key in enumerate(live_keys):
                            world_key = key[0]
                            policy_data['history_dicts'][key] = update_history(
                                history_dict=policy_data['history_dicts'][key],
                                partner_obs=partner_obs[idx],
                                log_history_step=policy_data['history']['log_history'],
                                current_step=policy_data['env_to_step_in_trial_dict'][f'world_{world_key}'],
                                trial=trial
                                )

                        live_worlds = list(set([idx[0] for idx in live_keys]))
                        for world in live_worlds:
                            policy_data['env_to_step_in_trial_dict'][f'world_{world}'] += 1 

                        
                    
            
            combined_mask = torch.zeros_like(live_agent_mask, dtype=torch.bool)
            for live_mask in policy_live_masks.values():
                combined_mask |= live_mask
            assert torch.all(live_agent_mask == combined_mask), "Live agent mask mismatch!"

            action_template = torch.zeros((num_worlds, max_agent_count), dtype=torch.int64, device=device)

            # Assign actions based on policy masks
            for policy_name, action in actions.items():
                live_mask = policy_live_masks[policy_name]
                if action.numel() > 0:
                    action_template[live_mask] = action.to(dtype=action_template.dtype, device=device)
                    
            assert(torch.all(action_template *combined_mask == action_template)), "mismatch between action template and combined mask" 
            # Step environment
            env.step_dynamics(action_template)

            if render_sim_state and len(active_worlds) > 0:
                    
                    has_live_agent = torch.where(
                        live_agent_mask[active_worlds, :].sum(axis=1) > 0
                    )[0].tolist()

                    if time_step % render_every_n_steps == 0:
                        if center_on_ego:
                            agent_indices = torch.argmax(control_mask.to(torch.uint8), dim=1).tolist()
                        else:
                            agent_indices = None

                        sim_state_figures = env.vis.plot_simulator_state(
                            env_indices=has_live_agent,
                            time_steps=[time_step] * len(has_live_agent),
                            zoom_radius=zoom_radius,
                            center_agent_indices=agent_indices,
                            policy_masks=policies 
                        )
                        for idx, env_id in enumerate(has_live_agent):
                            sim_state_frames[env_id].append(
                                img_from_fig(sim_state_figures[idx])
                            )


            next_obs = env.get_obs()
            dones = env.get_dones().bool()
            infos = env.get_infos()
            reward =env.get_rewards(
                collision_weight=collision_weight,
                off_road_weight=off_road_weight,
                goal_achieved_weight=goal_achieved_weight,
            )


            for policy_name, live_mask in policy_live_masks.items():
                policy_metrics[policy_name][trial]["off_road"][live_mask] += infos.off_road[live_mask]
                policy_metrics[policy_name][trial]["collided"][live_mask] += infos.collided[live_mask]
                policy_metrics[policy_name][trial]["goal_achieved"][live_mask] += infos.goal_achieved[live_mask]
                policy_metrics[policy_name][trial]["reward"][live_mask] += reward[live_mask]


            live_agent_mask[dones] = False

            # Process completed worlds
            num_dones_per_world = (dones & control_mask).sum(dim=1)
            total_controlled_agents = control_mask.sum(dim=1)
            done_worlds = (num_dones_per_world == total_controlled_agents).nonzero(as_tuple=True)[0]

            for world in done_worlds:
                if world in active_worlds:
                    active_worlds.remove(world)
                    episode_lengths[world] = time_step

            if return_agent_positions:
                global_agent_states = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor())
                agent_positions[:, :, time_step, 0] = global_agent_states.pos_x
                agent_positions[:, :, time_step, 1] = global_agent_states.pos_y

            if not active_worlds:  
                break               
    
    metrics =compute_metrics(policy_metrics,policies)

    #print(f"metrics are {metrics}")

    if render_sim_state:
        return metrics, sim_state_frames
    
    return metrics
