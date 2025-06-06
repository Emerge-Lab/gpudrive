import torch
import pandas as pd
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState

def multi_policy_rollout(
    env,
    policies, 
    device,
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
            env.max_cont_agents,  # This should be 64 to match your mask
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


    
    next_obs = env.reset()

    if reward_conditioning_present:
        reward_conditioned_obs = env.get_obs(get_reward_conditioned = True)

    policy_metrics = {
        policy_name: {
            trial: {
                "goal_achieved": torch.zeros((num_worlds, max_agent_count), device=device),
                "collided": torch.zeros((num_worlds, max_agent_count), device=device),
                "off_road": torch.zeros((num_worlds, max_agent_count), device=device),
                "reward": torch.zeros((num_worlds, max_agent_count), device=device),
            } for trial in range(k_trials)
        } for policy_name in policies
    }

    for trial in range(k_trials):

        episode_lengths = torch.zeros(num_worlds)
        
        active_worlds = list(range(num_worlds))
        control_mask = env.cont_agent_mask
        live_agent_mask = control_mask.clone()


        for time_step in range(episode_len):
            print(f't: {time_step}')

            policy_live_masks = {name: policy_data['mask'] & live_agent_mask for name, policy_data in policies.items()}


            actions = {}
            for policy_name, policy_data in policies.items():
                live_mask = policy_live_masks[policy_name]
                if live_mask.any():
                    if 'reward_conditioning' not in policy_data:
                        actions[policy_name], _, _, _ = policy_data['func'](
                            next_obs[live_mask], deterministic=deterministic
                        )
                    else:
                        obs = reward_conditioned_obs[live_mask]
                        actions[policy_name], _, _, _   = policy_data['func'](
                            obs, deterministic=deterministic
                        )

            
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
            
            if reward_conditioning_present:         
                reward_conditioned_obs = env.get_obs(get_reward_conditioned=True)
                

                
            dones = env.get_dones().bool()
            infos = env.get_infos()

     


            for policy_name, live_mask in policy_live_masks.items():
                policy_metrics[policy_name][trial]["off_road"][live_mask] += infos.off_road[live_mask]
                policy_metrics[policy_name][trial]["collided"][live_mask] += infos.collided[live_mask]
                policy_metrics[policy_name][trial]["goal_achieved"][live_mask] += infos.goal_achieved[live_mask]
                if 'reward_conditioning' not in policy_data:
                    reward =env.get_rewards(
                    collision_weight=policy_data['weights']['collision_weight'],
                    off_road_weight=policy_data['weights']['off_road_weight'],
                    goal_achieved_weight=policy_data['weights']['goal_achieved_weight'],
                    )
                    
                else:
                    reward = env.get_rewards(get_reward_conditioned = True)

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
        




    metrics =compute_metrics(policy_metrics,policies,k_trials)


    if render_sim_state:
        return metrics, sim_state_frames
    
    return metrics

def compute_metrics(policy_metrics,policies,k_trials):
        for policy_name, policy_data in policies.items():
            for trial in range(k_trials):
                
                controlled_per_scene = policy_data['mask'].sum(dim=1)
                
                policy_metrics[policy_name][trial]['off_road_count'] = ( policy_metrics[policy_name][trial]["off_road"] > 0).float().sum(axis=1)
                policy_metrics[policy_name][trial]['collided_count'] = (policy_metrics[policy_name][trial]["collided"]   > 0).float().sum(axis=1)
                policy_metrics[policy_name][trial]['goal_achieved_count'] = (policy_metrics[policy_name][trial]['goal_achieved']   > 0).float().sum(axis=1)
                
                policy_metrics[policy_name][trial]['frac_off_road'] = policy_metrics[policy_name][trial]['off_road_count'] / controlled_per_scene
                policy_metrics[policy_name][trial]['frac_collided'] = policy_metrics[policy_name][trial]['collided_count'] / controlled_per_scene
                policy_metrics[policy_name][trial]['frac_goal_achieved'] = policy_metrics[policy_name][trial]['goal_achieved_count'] / controlled_per_scene
            

        return policy_metrics



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
