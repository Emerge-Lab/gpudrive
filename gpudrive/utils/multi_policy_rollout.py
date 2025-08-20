import torch
import pandas as pd
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState
from box import Box
from gpudrive.utils.co_players import HistoryAgent, ReliableAgent, RewardConditionedAgent
from collections import defaultdict
from tqdm import tqdm
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.networks.history_late_fusion import NeuralNetWithHistory
from gpudrive.networks.late_fusion import NeuralNet
import math



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

    num_worlds = env.num_worlds
    max_agent_count = env.max_agent_count
    episode_len = env.config.episode_len
    sim_state_frames = {env_id: [] for env_id in range(num_worlds)}
    agent_positions = torch.zeros((num_worlds, max_agent_count, episode_len, 2))
   

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

    for trial in range(k_trials):
        next_obs = env.reset()
        episode_lengths = torch.zeros(num_worlds)
        
        active_worlds = list(range(num_worlds))
        control_mask = env.cont_agent_mask
        live_mask = control_mask.clone()

        world_time_steps = {
                f"world_{i}": 0 for i in range(num_worlds)
                }

        for time_step in range(episode_len):
            print(f't: {time_step}')



            action_template = torch.zeros((num_worlds, max_agent_count), dtype=torch.int64, device=device)

            for policy_name,policy in policies.items():

                action,_,_,_ = policy.get_action(next_obs,env)
                action_template[ policy.mask] = action.to(dtype=action_template.dtype, device=device)
                if hasattr(policy,'vectorized_history'):
                        policy.update_history(env, trial, world_time_steps)
                        
               
        
            if live_mask.any():
                live_indices = torch.nonzero(live_mask).view(-1, live_mask.dim())
                live_keys = [tuple(indice.tolist()) for indice in live_indices]
                live_worlds = list(set([idx[0] for idx in live_keys]))
                for world in live_worlds:
                    world_time_steps[f'world_{world}'] += 1 




            env.step_dynamics(action_template)

            if render_sim_state and len(active_worlds) > 0:
                    
                    has_live_agent = torch.where(
                        live_mask[active_worlds, :].sum(axis=1) > 0
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

                
            dones = env.get_dones().bool()
            infos = env.get_infos()
 

            for policy_name,policy in policies.items():
                policy_live_mask = policy.mask & live_mask
                policy_metrics[policy_name][trial]["off_road"][ policy_live_mask] += infos.off_road[policy_live_mask]
                policy_metrics[policy_name][trial]["collided"][ policy_live_mask] += infos.collided[policy_live_mask]
                policy_metrics[policy_name][trial]["goal_achieved"][policy_live_mask] += infos.goal_achieved[policy_live_mask]
                if not hasattr(policy,'reward_tensor'):
                    agent_reward =env.get_rewards(
                    collision_weight=policy.collision_weight,
                    off_road_weight=policy.off_road_weight,
                    goal_achieved_weight=policy.goal_achieved_weight,
                    )
                    
                else:
                    agent_reward = env.get_rewards(reward_tensor = policy.reward_tensor )

                
                real_reward = env.get_rewards(
                    collision_weight = env_collision_weight,
                    off_road_weight = env_off_road_weight,
                    goal_achieved_weight = env_goal_achieved_weight

                )

                if env_collision_weight == policy.collision_weight and env_off_road_weight == policy.off_road_weight and env_goal_achieved_weight == policy.goal_achieved_weight:
                     if hasattr(policy,'vectorized_history'):
                        assert torch.all(real_reward == agent_reward), "Real reward and agent reward should be equal if weights match"

                policy_metrics[policy_name][trial]["agent_reward"][policy_live_mask] += agent_reward[policy_live_mask]
                policy_metrics[policy_name][trial]["real_reward"][policy_live_mask] += real_reward[policy_live_mask]

            live_mask[dones] = False

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

            next_obs = env.get_obs()
        


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
    for policy_name, policy in policies.items():
        for trial in range(trials):
            controlled_mask = policy.mask
            num_controlled = controlled_mask.sum()          

            policy_metrics[policy_name][trial]['off_road'] = (policy_metrics[policy_name][trial]["off_road"][controlled_mask]).float()
            policy_metrics[policy_name][trial]['collided'] = (policy_metrics[policy_name][trial]["collided"][controlled_mask]).float()
            policy_metrics[policy_name][trial]['goal_achieved'] =( policy_metrics[policy_name][trial]['goal_achieved'][controlled_mask]).float()
            policy_metrics[policy_name][trial]['agent_reward'] =( policy_metrics[policy_name][trial]['agent_reward'][controlled_mask]).float()
            policy_metrics[policy_name][trial]['real_reward'] = (policy_metrics[policy_name][trial]['real_reward'][controlled_mask]).float()

            policy_metrics[policy_name][trial]['num_controlled'] = num_controlled 


            

    return policy_metrics





def configure_policies(policies_list,  policy_mask, device, num_worlds):
    """
    Configure policies from a list of policy dictionaries using agent classes.
    
    Args:
        policies_list: List of policy dictionaries from YAML
        policy_funcs: Dictionary mapping policy names to their functions
        policy_mask: Dictionary containing masks for each policy
        device: Device to run policies on

    """
    def create_reward_tensor(policy):
        reward_weights = {
                'collision_weight': policy.get('collision_weight', 0.0),
                'goal_achieved_weight': policy.get('goal_achieved_weight', 0.0),
                'off_road_weight': policy.get('off_road_weight', 0.0)
            }
            
       
        reward_tensor = torch.ones(
            num_worlds,
            64,  
            3,  
            device=device,
        )

        reward_tensor[:, :, 0] *= reward_weights['collision_weight']
        reward_tensor[:, :, 1] *= reward_weights['goal_achieved_weight']
        reward_tensor[:, :, 2] *= reward_weights['off_road_weight']

        return reward_tensor,reward_weights



    agents = {}
    
    for i, policy in enumerate(policies_list, 1):
        policy_key = f'pi_{i}'
        policy_name = policy['name']
        mask = policy_mask.get(policy_key)
        
  
        
        # Determine agent type and create appropriate agent
        if 'history' in policy:
            history_config = policy['history']
            episode_len = history_config.get( 'episode_len',91)
            log_history_step = history_config.get('log_history_step', 10)
            trials = history_config.get('k_trials', 5)
            partner_obs_shape = history_config.get('partner_obs_shape',378)
            closest_k_partners_in_history = history_config.get('closest_k_partners_in_history','all')

            reward_weights = {
            'collision_weight': policy.get('collision_weight', 0.0),
            'goal_achieved_weight': policy.get('goal_achieved_weight', 0.0),
            'off_road_weight': policy.get('off_road_weight', 0.0)
            }

            if hasattr(policy, "policy"):
                policy_func = policy.policy
            else:
                policy_func_data = torch.load(policy['file'], map_location=device, weights_only=False)
                
                policy_func = NeuralNetWithHistory( 
                input_dim=policy_func_data["model_arch"]["input_dim"],
                action_dim=policy_func_data["action_dim"],
                hidden_dim=policy_func_data["model_arch"]["hidden_dim"],
                k_trials=trials,
                num_steps=episode_len,
                log_history = log_history_step,
                closest_k_partners_in_history=closest_k_partners_in_history,
                )  

                policy_func.load_state_dict(policy_func_data['parameters'])
                policy_func.to(device)
    
            agents[policy_key] = HistoryAgent(
                name=policy_name,
                policy=policy_func,
                mask=mask,
                episode_len=episode_len,
                log_history_step=log_history_step,
                partner_obs_shape=partner_obs_shape,
                k_trials=trials,
                device=device,
                closest_k_partners_in_history =closest_k_partners_in_history,
                collision_weight=reward_weights['collision_weight'],
                goal_achieved_weight=reward_weights['goal_achieved_weight'],
                off_road_weight=reward_weights['off_road_weight'],
            )
            
        elif policy.get('reward_conditioning', False):
     

            policy_func_data = torch.load(policy['file'], map_location=device, weights_only=False)
            reward_tensor, reward_weights = create_reward_tensor(policy)
            config = Box()
            config['reward_type'] = 'reward_conditioned'

            policy_func = NeuralNet(
            input_dim=policy_func_data["model_arch"]["input_dim"],
            action_dim=policy_func_data["action_dim"],
            hidden_dim=policy_func_data["model_arch"]["hidden_dim"],
            config = config
            )

            policy_func.load_state_dict(policy_func_data['parameters'])
            policy_func.to(device)
            
            
            agents[policy_key] = RewardConditionedAgent(
                name=policy_name,
                policy=policy_func,
                mask=mask,
                reward_tensor=reward_tensor,
                device=device,
                collision_weight=reward_weights['collision_weight'],
                goal_achieved_weight =  reward_weights['goal_achieved_weight'],
                off_road_weight= reward_weights['off_road_weight']
            )

            
        else:
            if policy['name'] == 'reliable agent':
                policy_func = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27").to(device)
            else:
                config = Box()
                config['reward_type'] = 'weighted_combination'
                policy_func_data = torch.load(policy['file'], map_location=device, weights_only=False)
                policy_func = NeuralNet(
                input_dim=policy_func_data["model_arch"]["input_dim"],
                action_dim=policy_func_data["action_dim"],
                hidden_dim=policy_func_data["model_arch"]["hidden_dim"],
                config = config
                )

                policy_func.load_state_dict(policy_func_data['parameters'])
                policy_func.to(device)


            agents[policy_key] = ReliableAgent(
                name=policy_name,
                policy=policy_func,
                mask=mask,
                device=device
            )
    
    return agents


def create_policy_masks(env, num_worlds=10):
    policy_mask = torch.zeros_like(env.cont_agent_mask, dtype=torch.int)
    agent_indices = env.cont_agent_mask.nonzero(as_tuple=True)

    for world_idx, agent_idx in zip(*agent_indices):
        policy_mask[world_idx, agent_idx] = 2

    for world_idx in range(num_worlds):
        world_agents = (agent_indices[0] == world_idx)
        if world_agents.any():
            world_agent_indices = agent_indices[1][world_agents]
            random_agent_idx = world_agent_indices[torch.randint(0, len(world_agent_indices), (1,))]
            policy_mask[world_idx, random_agent_idx] = 1

    policy_masks = {
        'pi_1': (policy_mask == 1).reshape(num_worlds, -1),
        'pi_2': (policy_mask == 2).reshape(num_worlds, -1)
    }
    
    return policy_masks


def generate_crossplay_data(env, data_loader, config, device):


    policies = config.policies

    world_iterations = getattr(config, 'world_iterations', 1)
    
    df = defaultdict(lambda: defaultdict(float)) 
    agent_results = {}
    batch_count = 0

    for batch in tqdm(
        data_loader,
        desc=f"Processing batches",
        total=len(data_loader),
        colour="blue",
    ):
        batch_count += 1 
        env.swap_data_batch(batch)

        for world_iter in range(world_iterations):
            print(f"\nBatch {batch_count}, World Iteration {world_iter + 1}/{world_iterations}")
            
            policy_mask = create_policy_masks(env, config.num_worlds)

            for policy1 in policies:
                for policy2 in policies:

                    for trial in range(config.k_trials):
                        if trial not in agent_results:
                            agent_results[trial] = {}
                        key = (policy1['name'], policy2['name'])
                        if key not in agent_results[trial]:
                            agent_results[trial][key] = {}

                    agents = configure_policies([policy1, policy2], policy_mask, device, config.num_worlds)

                    policy_metrics = multi_policy_rollout(
                        env,
                        agents,
                        device,
                        k_trials=config.k_trials,
                        env_collision_weight=config.collision_weight,
                        env_off_road_weight=config.off_road_weight,
                        env_goal_achieved_weight=config.goal_achieved_weight
                    )

                    metrics = policy_metrics['pi_1']
                    key = (policy1['name'], policy2['name'])
                    for trial, metric_data in metrics.items():
                        print(f"  World {world_iter+1}, Trial: {trial}, Policies: {policy1['name']} vs {policy2['name']}")
                        
                        for metric_name, metric_value in metric_data.items():
                            val_type = type(metric_value).__name__
                            if isinstance(metric_value, torch.Tensor):
                                val_shape = metric_value.shape
                                val_preview = metric_value.flatten()[:3].tolist() if metric_value.numel() > 0 else []
                                print(f"    {metric_name}: {val_type} {val_shape} {val_preview}...")
                            else:
                                print(f"    {metric_name}: {val_type} {metric_value}")

                            if metric_name not in agent_results[trial][key]:
                                if isinstance(metric_value, torch.Tensor):
                                    agent_results[trial][key][metric_name] = metric_value.to(device).clone()
                                    print(f"      → Initialized tensor")
                                else:
                                    agent_results[trial][key][metric_name] = torch.tensor(metric_value, device=device)
                                    print(f"      → Initialized from value")
                            else:
                                if isinstance(metric_value, torch.Tensor):
                                    agent_results[trial][key][metric_name] += metric_value.to(device)
                                    print(f"      → Added to existing tensor")
                                else:
                                    agent_results[trial][key][metric_name] += torch.tensor(metric_value, device=device)
                                    print(f"      → Added to existing from value")

                            val = metric_value.sum().item() if isinstance(metric_value, torch.Tensor) and metric_value.ndim > 0 else float(metric_value)
                            df[(trial, policy1['name'], policy2['name'])][metric_name] += val
                            print(f"      → DF aggregated: {val}")
                        
                        print(f"    Current DF state: {dict(df[(trial, policy1['name'], policy2['name'])])}")
                        print("-" * 30)


        for policy1 in policies:
            for policy2 in policies:
                key = (policy1['name'], policy2['name'])
                for trial in range(config.k_trials):
                    if trial in agent_results and key in agent_results[trial]:
                        data = agent_results[trial][key]
                        df_key = (trial, policy1['name'], policy2['name'])

                        if 'goal_achieved' in data and data['goal_achieved'].sum().item() > 0:
                            df[df_key]['frac_goal_achieved'] = float(data['goal_achieved'].sum().item() / data['num_controlled'].sum().item())
                            n = data['goal_achieved'].numel()
                            df[df_key]['frac_goal_achieved_se'] = (data['goal_achieved'].std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0

                        if 'off_road' in data:
                            num_controlled = data['num_controlled'].sum().item()
                            total_off_road = data['off_road'].sum().item()
                            n = data['off_road'].numel()
                            
                            df[df_key]['frac_off_road'] = float(total_off_road / num_controlled) if num_controlled > 0 else 0.0
                            df[df_key]['frac_off_road_se'] = (data['off_road'].std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0

                        if 'collided' in data:
                            num_controlled = data['num_controlled'].sum().item()
                            total_collided = data['collided'].sum().item()
                            n = data['collided'].numel()
                            
                            df[df_key]['frac_collided'] = float(total_collided / num_controlled) if num_controlled > 0 else 0.0
                            df[df_key]['frac_collided_se'] = (data['collided'].std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0

                        if 'agent_reward' in data:
                            df[df_key]['frac_agent_reward'] = float(data['agent_reward'].sum().item() / data['num_controlled'].sum().item())
                            n = data['agent_reward'].numel()
                            df[df_key]['frac_agent_reward_se'] = (data['agent_reward'].std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0

                        if 'real_reward' in data:
                            df[df_key]['frac_real_reward'] = float(data['real_reward'].sum().item() / data['num_controlled'].sum().item())
                            n = data['real_reward'].numel()
                            df[df_key]['frac_real_reward_se'] = (data['real_reward'].std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0

                        df[df_key]['num_controlled'] += data['num_controlled'].sum().item()
        max_test_set_iterations = config.get("max_test_set_iterations", float('inf'))
        if max_test_set_iterations >= batch_count:
            break

    
    return df, agent_results, policies


def generate_history_agent_data(env, data_loader, config, device='cuda'):

    policies = config.policies
    
    # history_policies = [p for p in policies if p['name'].lower() == 'history']
    history_policies = []
    for p in policies:
        if p['name'].lower() == 'history':
            history_policies.append(p)
    
    if len(history_policies) == 0:
        raise ValueError("History policy not found in policies list")
    elif len(history_policies) > 1:
        raise ValueError("More than 1 history agent, use crossplay evaluation if you wish to compare different history agents with one another")
    
    history_policy = history_policies[0]

    df = defaultdict(lambda: defaultdict(float))
    agent_results = {}

    for batch in tqdm(
        data_loader,
        desc=f"Processing batches",
        total=len(data_loader),
        colour="blue",
    ):

        env.swap_data_batch(batch)

        policy_mask = create_policy_masks(env, config.num_worlds)

        for policy2 in policies:
            if config.get("skip_history", True) and policy2['name'].lower() == "history":
                continue
            for trial in range(config.k_trials):
                if trial not in agent_results:
                    agent_results[trial] = {}
                key = (history_policy['name'], policy2['name'])
                if key not in agent_results[trial]:
                    agent_results[trial][key] = {}

            agents = configure_policies([history_policy, policy2], policy_mask, device, config.num_worlds)

            policy_metrics = multi_policy_rollout(
                env,
                agents,
                device,
                k_trials=config.k_trials,
                env_collision_weight=config.collision_weight,
                env_off_road_weight=config.off_road_weight,
                env_goal_achieved_weight=config.goal_achieved_weight
            )


            metrics = policy_metrics['pi_1']
            key = (history_policy['name'], policy2['name'])

            for trial, metric_data in metrics.items():
                print(f"Trial: {trial}")
                
                for metric_name, metric_value in metric_data.items():

                    val_type = type(metric_value).__name__
                    if isinstance(metric_value, torch.Tensor):
                        val_shape = metric_value.shape
                        val_preview = metric_value.flatten()[:3].tolist() if metric_value.numel() > 0 else []


                    
                    if metric_name not in agent_results[trial][key]:
                        if isinstance(metric_value, torch.Tensor):
                            agent_results[trial][key][metric_name] = metric_value.to(device).clone().unsqueeze(0)

                        else:
                            agent_results[trial][key][metric_name] = torch.tensor([metric_value], device=device)
                    else:
                        if isinstance(metric_value, torch.Tensor):
                            new_val = metric_value.to(device).unsqueeze(0)
                            agent_results[trial][key][metric_name] = torch.cat([agent_results[trial][key][metric_name], new_val])

                        else:
                            new_val = torch.tensor([metric_value], device=device)
                            agent_results[trial][key][metric_name] = torch.cat([agent_results[trial][key][metric_name], new_val])

                    val = metric_value.sum().item() if isinstance(metric_value, torch.Tensor) and metric_value.ndim > 0 else float(metric_value)
                    df[(trial, history_policy['name'], policy2['name'])][metric_name] += val




    for policy2 in policies:
        key = (history_policy['name'], policy2['name'])
        print(f"num trials {config.k_trials}")
        for trial in range(config.k_trials):
            if trial in agent_results and key in agent_results[trial]:
                data = agent_results[trial][key]
                df_key = (trial, history_policy['name'], policy2['name'])

                if 'goal_achieved' in data and data['goal_achieved'].sum().item() > 0:
                    df[df_key]['frac_goal_achieved'] = float(data['goal_achieved'].sum().item() / data['num_controlled'].sum().item())
                    n = data['goal_achieved'].numel()
                    df[df_key]['frac_goal_achieved_se'] = (data['goal_achieved'].std(unbiased=True) / torch.sqrt(torch.tensor(float(n)))).item() if n > 1 else 0.0

                if 'off_road' in data and data['off_road'].sum().item() > 0:
                    df[df_key]['frac_off_road'] = float(data['off_road'].sum().item() / data['num_controlled'].sum().item())
                    n = data['off_road'].numel()
                    df[df_key]['frac_off_road_se'] = (data['off_road'].std(unbiased=True) / torch.sqrt(torch.tensor(float(n)))).item() if n > 1 else 0.0

                if 'collided' in data and data['collided'].sum().item() > 0:
                    df[df_key]['frac_collided'] = float(data['collided'].sum().item() / data['num_controlled'].sum().item())
                    n = data['collided'].numel()
                    df[df_key]['frac_collided_se'] = (data['collided'].std(unbiased=True) / torch.sqrt(torch.tensor(float(n)))).item() if n > 1 else 0.0

                if 'agent_reward' in data and data['agent_reward'].sum().item() > 0:
                    df[df_key]['frac_agent_reward'] = float(data['agent_reward'].sum().item() / data['num_controlled'].sum().item())
                    n = data['agent_reward'].numel()
                    df[df_key]['frac_agent_reward_se'] = (data['agent_reward'].std(unbiased=True) / torch.sqrt(torch.tensor(float(n)))).item() if n > 1 else 0.0

                if 'real_reward' in data and data['real_reward'].sum().item() > 0:
                    df[df_key]['frac_real_reward'] = float(data['real_reward'].sum().item() / data['num_controlled'].sum().item())
                    n = data['real_reward'].numel()
                    df[df_key]['frac_real_reward_se'] = (data['real_reward'].std(unbiased=True) / torch.sqrt(torch.tensor(float(n)))).item() if n > 1 else 0.0

                df[df_key]['num_controlled'] += data['num_controlled'].sum().item()

    return df, policies, history_policy



def generate_human_log_data(env,data_loader, config, device):

        data_table = {}
        for policy in config.policies:

            policy_masks = {'pi_1':env.cont_agent_mask}
            policy_set=configure_policies([policy],policy_masks,config.device, config.num_worlds)
            metrics = multi_policy_rollout(env,policy_set,device,k_trials=config.k_trials)
            metrics = metrics['pi_1']

            for trial, metric_data in metrics.items():
                # Compute fractions from raw counts like version B
                num_controlled = metric_data['num_controlled']
                
                # Calculate fractions by dividing by num_controlled
                frac_goal_achieved = 0.0
                frac_goal_achieved_std = 0.0
                if 'goal_achieved' in metric_data and metric_data['goal_achieved'].sum().item() > 0:
                    goal_achieved = metric_data['goal_achieved']
                    frac_goal_achieved = float(goal_achieved.sum().item() / num_controlled.sum().item())
                    frac_goal_achieved_std = (goal_achieved.std(unbiased=True) / (goal_achieved.numel() ** 0.5)).item() if goal_achieved.numel() > 1 else 0.0
                
                frac_collided = 0.0
                frac_collided_std = 0.0
                if 'collided' in metric_data and metric_data['collided'].sum().item() > 0:
                    collided = metric_data['collided']
                    frac_collided = float(collided.sum().item() / num_controlled.sum().item())
                    frac_collided_std = (collided.std(unbiased=True) / (collided.numel() ** 0.5)).item() if collided.numel() > 1 else 0.0
                
                frac_off_road = 0.0
                frac_off_road_std = 0.0
                if 'off_road' in metric_data and metric_data['off_road'].sum().item() > 0:
                    off_road = metric_data['off_road']
                    frac_off_road = float(off_road.sum().item() / num_controlled.sum().item())
                    frac_off_road_std = (off_road.std(unbiased=True) / (off_road.numel() ** 0.5)).item() if off_road.numel() > 1 else 0.0
                
                frac_agent_reward = 0.0
                frac_agent_reward_std = 0.0
                if 'agent_reward' in metric_data and metric_data['agent_reward'].sum().item():

                    agent_reward = metric_data['agent_reward']
                    frac_agent_reward = float(agent_reward.sum().item() / num_controlled.sum().item())
                    frac_agent_reward_std = (agent_reward.std(unbiased=True) / (agent_reward.numel() ** 0.5)).item() if agent_reward.numel() > 1 else 0.0
                
                frac_real_reward = 0.0
                frac_real_reward_std = 0.0
                if 'real_reward' in metric_data and metric_data['real_reward'].sum().item():
                    real_reward = metric_data['real_reward']
                    frac_real_reward = float(real_reward.sum().item() / num_controlled.sum().item())
                    frac_real_reward_std = (real_reward.std(unbiased=True) / (real_reward.numel() ** 0.5)).item() if real_reward.numel() > 1 else 0.0
                
                # Store computed fractions in data_table
                data_table[(trial, policy['name'])] = {
                    "frac_goal_achieved": frac_goal_achieved,
                    "frac_collided": frac_collided,
                    "frac_off_road": frac_off_road,
                    "frac_agent_reward": frac_agent_reward,
                    "frac_real_reward": frac_real_reward,
                    'frac_off_road_std': frac_off_road_std,
                    'frac_collided_std': frac_collided_std,
                    'frac_goal_achieved_std': frac_goal_achieved_std,
                    'frac_agent_reward_std': frac_agent_reward_std,
                    'frac_real_reward_std': frac_real_reward_std,
                }


        return data_table