import torch
import pandas as pd
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState

def multi_policy_rollout(
    env,
    policies, 
    device,
    env_collision_weight: float = -0.75,
    env_off_road_weight: float = -0.75,
    env_goal_achieved_weight: float =1, 
    deterministic: bool = False,
    render_sim_state: bool = False,
    render_every_n_steps: int = 1,
    zoom_radius: int = 100,
    return_agent_positions: bool = False,
    center_on_ego: bool = False,
    k_trials:int =1,
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
    
    # Initialize storage
    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len
    sim_state_frames = {env_id: [] for env_id in range(num_worlds)}
    agent_positions = torch.zeros((num_worlds, max_agent_count, episode_len, 2))

    reward_conditioning_present = False

    for policy_name, policy_data in policies.items():
        if 'reward_conditioning' in policy_data:
            reward_conditioning_present = True


    if reward_conditioning_present:
        env.reward_weights_tensor = torch.zeros(
            env.num_worlds,
            64,  # This should be 64 to match your mask
            3,  # collision, goal_achieved, off_road
            device=env.device,
        )
        for policy_name, policy_data in policies.items():
            if 'reward_conditioning' in policy_data:
                mask = policy_data['mask'] 

                collision_weight = torch.tensor(policy_data['reward_conditioning']['collision_weight'], device=env.device)
                goal_weight = torch.tensor(policy_data['reward_conditioning']['goal_achieved_weight'], device=env.device)
                offroad_weight = torch.tensor(policy_data['reward_conditioning']['off_road_weight'], device=env.device)

                collision_tensor = mask * collision_weight
                goal_tensor = mask * goal_weight
                offroad_tensor = mask * offroad_weight

                env.reward_weights_tensor[:, :, 0] += collision_tensor
                env.reward_weights_tensor[:, :, 1] += goal_tensor
                env.reward_weights_tensor[:, :, 2] += offroad_tensor


    
    

    if reward_conditioning_present:
        reward_conditioned_obs = env.get_obs(get_reward_conditioned = True)

    policy_metrics = {
        policy_name: {
            trial: {
                "goal_achieved": torch.zeros((num_worlds, max_agent_count), device=device),
                "collided": torch.zeros((num_worlds, max_agent_count), device=device),
                "off_road": torch.zeros((num_worlds, max_agent_count), device=device),
                "agent_reward":  torch.zeros((num_worlds, max_agent_count), device=device),
                "real_reward": torch.zeros((num_worlds, max_agent_count), device=device),
            } for trial in range(k_trials)
        } for policy_name in policies
    }

    all_partner_observations = env._get_partner_obs()
    partner_obs_dim = all_partner_observations.shape[2]

    for policy,policy_data in policies.items():
            if 'history' in policy_data:
                controlled_indices = torch.nonzero(policy_data['mask']).squeeze()
                controlled_keys = [tuple(indice.tolist()) for indice in controlled_indices]
                policy_data['history_dicts'] ={}
                for keys in controlled_keys:
                    policy_data['history_dicts'][keys] =initialise_history(policy_data['history']['trials'],
                                                                        policy_data['history']['num_steps'],
                                                                    policy_data['history']['log_history'],
                                                                    partner_obs_dim,
                                                                    device)
  

    
    for trial in range(k_trials):
        next_obs = env.reset()
        episode_lengths = torch.zeros(num_worlds)
        
        active_worlds = list(range(num_worlds))
        control_mask = env.cont_agent_mask
        live_agent_mask = control_mask.clone()

        world_time_steps = {
                f"world_{i}": 0 for i in range(num_worlds)
                }

        for time_step in range(episode_len):
            print(f't: {time_step}')

            policy_live_masks = {name: policy_data['mask'] & live_agent_mask for name, policy_data in policies.items()}


            actions = {}
            for policy_name, policy_data in policies.items():
                live_mask = policy_live_masks[policy_name]
                if live_mask.any():
                    if 'reward_conditioning'  in policy_data:

                        actions[policy_name], _, _, _   = policy_data['func'](
                            reward_conditioned_obs[live_mask], deterministic=deterministic
                        )

                    elif 'history' in policy_data:
                        live_observations = next_obs[live_mask]
                        live_indices = torch.nonzero(live_mask).view(-1, live_mask.dim())
                        live_keys = [tuple(indice.tolist()) for indice in live_indices]

                        observations_list = []
                        for idx, key in enumerate(live_keys):
                            world_key = key[0]
                            agent_key = key[1]
                            observation = next_obs[world_key][agent_key]
                            assert torch.all(observation == live_observations[idx]), "mismatch"
                            observation_with_history = get_history_batch(policy_data['history_dicts'][key],
                                                                policy_data['history']['trials'],
                                                                observation,
                                                                device)

                            observations_list.append(observation_with_history)

                        agent_observations = torch.vstack(observations_list)


                        actions[policy_name], _, _, _  = policy_data['func'](
                            agent_observations, deterministic=deterministic
                        )
                                    
                        for idx,key in enumerate(live_keys):
                            
                            world_key = key[0]
                            agent_key = key[1]
                            # assert torch.all(partner_obs[idx] == all_partner_observations[world_key][agent_key]), "mismatch"
                            policy_data['history_dicts'][key] = update_history(
                                history_dict=policy_data['history_dicts'][key],
                                partner_obs=all_partner_observations[world_key][agent_key],
                                log_history_step=policy_data['history']['log_history'],
                                current_step=world_time_steps[f'world_{world_key}'],
                                trial=trial
                                )
     

                    else:
                        actions[policy_name], _, _, _ = policy_data['func'](
                            next_obs[live_mask], deterministic=deterministic
                        )

        
            if live_agent_mask.any():
                live_indices = torch.nonzero(live_agent_mask).view(-1, live_agent_mask.dim())
                live_keys = [tuple(indice.tolist()) for indice in live_indices]
                live_worlds = list(set([idx[0] for idx in live_keys]))
                for world in live_worlds:
                    world_time_steps[f'world_{world}'] += 1 

            
            combined_mask = torch.zeros_like(live_agent_mask, dtype=torch.bool)
            for live_mask in policy_live_masks.values():
                combined_mask |= live_mask
            assert torch.all(live_agent_mask == combined_mask), "Live agent mask mismatch!"

            action_template = torch.zeros((num_worlds, max_agent_count), dtype=torch.int64, device=device) ## can condense and simplify this

            for policy_name, action in actions.items():
                live_mask = policy_live_masks[policy_name]
                if action.numel() > 0:
                    action_template[live_mask] = action.to(dtype=action_template.dtype, device=device)

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


            # Update observations and agent statuses
            next_obs = env.get_obs()
            if reward_conditioning_present:         
                reward_conditioned_obs = env.get_obs(get_reward_conditioned=True)
                
            all_partner_observations =  env._get_partner_obs()
                
            dones = env.get_dones().bool()
            infos = env.get_infos()

     


            for policy_name, live_mask in policy_live_masks.items():
                policy_metrics[policy_name][trial]["off_road"][live_mask] += infos.off_road[live_mask]
                policy_metrics[policy_name][trial]["collided"][live_mask] += infos.collided[live_mask]
                policy_metrics[policy_name][trial]["goal_achieved"][live_mask] += infos.goal_achieved[live_mask]
                if 'reward_conditioning' not in policy_data:
                    agent_reward =env.get_rewards(
                    collision_weight=policy_data['weights']['collision_weight'],
                    off_road_weight=policy_data['weights']['off_road_weight'],
                    goal_achieved_weight=policy_data['weights']['goal_achieved_weight'],
                    )
                    
                else:
                    agent_reward = env.get_rewards(get_reward_conditioned = True)
                
                real_reward = env.get_rewards(
                    collision_weight = env_collision_weight,
                    off_road_weight = env_off_road_weight,
                    goal_achieved_weight = env_goal_achieved_weight

                )

                policy_metrics[policy_name][trial]["agent_reward"][live_mask] += agent_reward[live_mask]
                policy_metrics[policy_name][trial]["real_reward"][live_mask] += real_reward[live_mask]

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
        




    metrics =compute_metrics(policy_metrics,policies,k_trials)


    if render_sim_state:
        return metrics, sim_state_frames
    
    return metrics




def create_data_table(data):
    # Extract unique policies
    policies = sorted(set(policy for pair in data.keys() for policy in pair))

    # Create empty DataFrames
    collisions_table = pd.DataFrame(index=policies, columns=policies)
    off_roads_table = pd.DataFrame(index=policies, columns=policies)
    goal_achieved_table = pd.DataFrame(index=policies, columns=policies)

    # Populate DataFrames
    for (p1, p2), metrics in data.items():
        collisions_table.loc[p1, p2] = metrics['frac_collided'].item()
        off_roads_table.loc[p1, p2] = metrics['frac_off_road'].item()
        goal_achieved_table.loc[p1, p2] = metrics['frac_goal_achieved'].item()

    # Print Tables
    print("Average Collisions Table:")
    print(collisions_table, "\n")

    print("Average Off Roads Table:")
    print(off_roads_table, "\n")

    print("Average Goal Achieved Table:")
    print(goal_achieved_table, "\n")

def compute_metrics(policy_metrics,policies,trials):
    for policy_name, policy_data in policies.items():
        for trial in range(trials):
            
            controlled_per_scene = policy_data['mask'].sum(dim=1)
            

            policy_metrics[policy_name][trial]['off_road_count'] = (policy_metrics[policy_name][trial]["off_road"] > 0).float().sum(axis=1)
            policy_metrics[policy_name][trial]['collided_count'] = (policy_metrics[policy_name][trial]["collided"] > 0).float().sum(axis=1)
            policy_metrics[policy_name][trial]['goal_achieved_count'] = (policy_metrics[policy_name][trial]['goal_achieved'] > 0).float().sum(axis=1)
            policy_metrics[policy_name][trial]['agent_reward_count'] = (policy_metrics[policy_name][trial]['agent_reward'] > 0).float().sum(axis=1)
            policy_metrics[policy_name][trial]['real_reward_count'] = (policy_metrics[policy_name][trial]['real_reward'] > 0).float().sum(axis=1)


            policy_metrics[policy_name][trial]['off_road_per_world'] = policy_metrics[policy_name][trial]['off_road_count'] / controlled_per_scene
            policy_metrics[policy_name][trial]['collided_per_world'] = policy_metrics[policy_name][trial]['collided_count'] / controlled_per_scene 
            policy_metrics[policy_name][trial]['goal_achieved_per_world'] = policy_metrics[policy_name][trial]['goal_achieved_count'] / controlled_per_scene
            policy_metrics[policy_name][trial]['agent_reward_per_world'] = policy_metrics[policy_name][trial]['agent_reward_count'] / controlled_per_scene
            policy_metrics[policy_name][trial]['real_reward_per_world'] = policy_metrics[policy_name][trial]['real_reward_count'] / controlled_per_scene

            policy_metrics[policy_name][trial]['frac_off_road'] = policy_metrics[policy_name][trial]['off_road_count'].sum() /  controlled_per_scene.sum()
            policy_metrics[policy_name][trial]['frac_collided'] = policy_metrics[policy_name][trial]['collided_count'].sum() /controlled_per_scene.sum()
            policy_metrics[policy_name][trial]['frac_goal_achieved'] = policy_metrics[policy_name][trial]['goal_achieved_count'].sum() / controlled_per_scene.sum()
            policy_metrics[policy_name][trial]['frac_agent_reward'] = policy_metrics[policy_name][trial]['agent_reward_count'].sum() / controlled_per_scene.sum()
            policy_metrics[policy_name][trial]['frac_real_reward'] = policy_metrics[policy_name][trial]['real_reward_count'].sum() / controlled_per_scene.sum()
            
            # Add standard deviations
            controlled_mask = policy_data['mask']

            off_road_mask = (policy_metrics[policy_name][trial]["off_road"] > 0)[controlled_mask]
            collided_mask = (policy_metrics[policy_name][trial]["collided"] > 0)[controlled_mask]
            goal_achieved_mask = (policy_metrics[policy_name][trial]["goal_achieved"] > 0)[controlled_mask]
            agent_reward_mask = (policy_metrics[policy_name][trial]["agent_reward"] > 0)[controlled_mask]
            real_reward_mask = (policy_metrics[policy_name][trial]["real_reward"] > 0)[controlled_mask]

            policy_metrics[policy_name][trial]['frac_off_road_std'] = off_road_mask.float().std()
            policy_metrics[policy_name][trial]['frac_collided_std'] = collided_mask.float().std()
            policy_metrics[policy_name][trial]['frac_goal_achieved_std'] = goal_achieved_mask.float().std()
            policy_metrics[policy_name][trial]['frac_agent_reward_std'] = agent_reward_mask.float().std()
            policy_metrics[policy_name][trial]['frac_real_reward_std'] = real_reward_mask.float().std()
            

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