import torch
from gpudrive.env import constants


class ReliableAgent:
    def __init__(self,name,policy,mask,device, collision_weight = -0.75, goal_achieved_weight =1 , off_road_weight = -0.75):
        self.name = name
        self.policy = policy
        self.mask = mask
        self.device = device
        self.collision_weight = collision_weight
        self.goal_achieved_weight = goal_achieved_weight
        self.off_road_weight = off_road_weight
    
    def get_action(self,observations, env):
        observations = observations[self.mask]
        return self.policy(observations)


class RewardConditionedAgent:
    def __init__(self, name, policy, mask, reward_tensor, device, collision_weight = -0.75, goal_achieved_weight =1 , off_road_weight = -0.75):
        self.name = name
        self.policy = policy
        self.reward_tensor = reward_tensor
        self.mask = mask
        self.device = device
        self.collision_weight = collision_weight
        self.goal_achieved_weight = goal_achieved_weight
        self.off_road_weight = off_road_weight
  

    def get_action(self,observations,env):
        observations = env.get_obs(reward_tensor=self.reward_tensor)[self.mask ]
        return self.policy(observations)
    
    
class HistoryAgent:
    def __init__(self, name, policy, mask, episode_len, log_history_step, partner_obs_shape, k_trials, device= "cuda",
                closest_k_partners_in_history='all',  collision_weight=-0.75, 
                goal_achieved_weight=1, off_road_weight=-0.75):
        self.name = name
        self.policy = policy
        self.mask = mask
        self.device = device
        self.k_trials = k_trials
        self.log_history_step = log_history_step
        self.episode_len = episode_len
        self.partner_obs_shape = partner_obs_shape

        self.closest_k_partners_in_history = closest_k_partners_in_history
        self.k = self.closest_k_partners_in_history if self.closest_k_partners_in_history != 'all' else None

        controlled_indices = torch.nonzero(self.mask).squeeze()
        self.controlled_keys = [tuple(indice.tolist()) for indice in controlled_indices]

        self.collision_weight = collision_weight
        self.goal_achieved_weight = goal_achieved_weight
        self.off_road_weight = off_road_weight

        self._build_vectorized_history_structure()

        self.trial_keys = [f"trial_{k}" for k in range(self.k_trials)]

    def _build_vectorized_history_structure(self):
        """
        Build vectorized history tensor for ultra-fast access.
        Shape: (n_agents, n_trials, n_steps, history_feature_size)
        """
        n_agents = len(self.controlled_keys)
        n_trials = self.k_trials
        n_steps = self.episode_len // self.log_history_step + 1
        
        if self.closest_k_partners_in_history == 'all':
            history_size = self.partner_obs_shape
        else:
            history_size = constants.PARTNER_FEAT_DIM * self.closest_k_partners_in_history

        self.vectorized_history = torch.zeros(
            n_agents, n_trials, n_steps, history_size,
            device=self.device, dtype=torch.float32
        )

        self.agent_to_vector_idx = {}
        for i, (world_key, agent_key) in enumerate(self.controlled_keys):
            self.agent_to_vector_idx[(world_key, agent_key)] = i

    def _get_keys(self, mask):
        """Extract live keys once and reuse across methods"""
        indices = torch.nonzero(mask, as_tuple=False).view(-1, mask.dim())
        return [tuple(idx.tolist()) for idx in indices]

    def update_history(self, env, trial, world_time_steps):
        """
        Vectorized history update using batch processing.
        """
        all_partner_observations = env._get_partner_obs()
        
        # Collect agents that need updates
        agents_to_process = []
        agent_indices = []
        
        for i, (world_key, agent_key) in enumerate(self.controlled_keys):
            current_step = world_time_steps[f'world_{world_key}']
            if current_step % self.log_history_step == 0:
                step_index = current_step // self.log_history_step
                agents_to_process.append((world_key, agent_key, trial, step_index, i))
                agent_indices.append(i)
        
        if not agents_to_process:
            return

        partner_obs_list = []
        for world_key, agent_key, trial, step_index, _ in agents_to_process:
            agents_partner_observations = all_partner_observations[world_key][agent_key]
            partner_obs_list.append(agents_partner_observations)

        batch_partner_obs = torch.stack(partner_obs_list, dim=0)

        batch_sorted = self.batch_sort_partner_obs(batch_partner_obs)

        vector_agent_indices = torch.zeros(len(agents_to_process), dtype=torch.long, device=self.device)
        trial_indices = torch.zeros(len(agents_to_process), dtype=torch.long, device=self.device)
        step_indices = torch.zeros(len(agents_to_process), dtype=torch.long, device=self.device)
        
        for i, (world_key, agent_key, trial, step_index, _) in enumerate(agents_to_process):
            vector_agent_indices[i] = self.agent_to_vector_idx[(world_key, agent_key)]
            trial_indices[i] = trial
            step_indices[i] = step_index

        flattened_sorted = batch_sorted.flatten(start_dim=1)
 
        self.vectorized_history[vector_agent_indices, trial_indices, step_indices] = flattened_sorted

    def batch_sort_partner_obs(self, batch_partner_obs):
        """
        Batch version of sort_partner_obs for multiple agents simultaneously.
        """
        batch_size = batch_partner_obs.shape[0]
        partner_obs_reshaped = batch_partner_obs.view(batch_size, 63, 6)

        non_partners = (partner_obs_reshaped.sum(dim=2) == 0)
        
        distances = torch.sqrt((partner_obs_reshaped[:, :, 1] - partner_obs_reshaped[:, :, 2]) ** 2)
        distances[non_partners] = float('inf')

        if self.k is not None and self.k < 63:
            _, indices = torch.topk(distances, self.k, dim=1, largest=False)
            batch_indices = torch.arange(batch_size, device=indices.device).unsqueeze(1).expand(-1, self.k)
            sorted_partners = partner_obs_reshaped[batch_indices, indices]
        else:
            # Sort all partners
            sorted_indices = torch.argsort(distances, dim=1)
            batch_indices = torch.arange(batch_size, device=sorted_indices.device).unsqueeze(1).expand(-1, 63)
            sorted_partners = partner_obs_reshaped[batch_indices, sorted_indices]
        
        return sorted_partners


    def get_obs_and_history(self, obs):
        """
        Vectorized version of observation and history retrieval.
        """

        
        live_keys = self._get_keys(self.mask)
        
        if not live_keys:
            return torch.tensor([])

        current_obs_list = []
        agent_indices = []
        
        for world_key, agent_key in live_keys:
            current_obs_list.append(obs[world_key][agent_key])
            agent_indices.append(self.agent_to_vector_idx[(world_key, agent_key)])

        current_obs_batch = torch.stack(current_obs_list, dim=0)

        agent_indices_tensor = torch.tensor(agent_indices, dtype=torch.long, device=self.device)
        agent_histories = self.vectorized_history[agent_indices_tensor]  
        flattened_histories = agent_histories.flatten(start_dim=1) 

        combined_obs = torch.cat([current_obs_batch, flattened_histories], dim=1)
        
        return combined_obs

    def get_action(self, observations_less_history, env):
        """
        Get action using vectorized observations and history.
        """
        observations = self.get_obs_and_history(observations_less_history)
        action = self.policy(observations)
        return action

    def clear_history(self, world_indices=None, clear_all=False):
        """
        Clear history for specified worlds or all agents.
        """
        if clear_all:
            self.vectorized_history.zero_()
        else:
            if world_indices is None:
                raise ValueError("world_indices must be provided when clear_all=False")
                
            if not isinstance(world_indices, torch.Tensor):
                world_indices = torch.tensor(world_indices, device=self.device)
            
            worlds_to_clear = set(world_indices.tolist())
            
            # Find agents in worlds to clear
            agents_to_clear = []
            for i, (world_key, agent_key) in enumerate(self.controlled_keys):
                if world_key in worlds_to_clear:
                    agents_to_clear.append(i)
            
            if agents_to_clear:
                agents_to_clear_tensor = torch.tensor(agents_to_clear, dtype=torch.long, device=self.device)
                self.vectorized_history[agents_to_clear_tensor] = 0.0

    def reset_trial_history(self, trial_idx):
        """
        Reset history for a specific trial across all agents.
        """
        if 0 <= trial_idx < self.k_trials:
            self.vectorized_history[:, trial_idx, :, :] = 0.0
        else:
            raise ValueError(f"Trial index {trial_idx} out of range [0, {self.k_trials-1}]")

    def get_history_stats(self):
        """
        Get statistics about the current history state.
        """
        non_zero_entries = torch.count_nonzero(self.vectorized_history)
        total_entries = self.vectorized_history.numel()
        memory_usage = self.vectorized_history.element_size() * total_entries / (1024 * 1024)  # MB
        
        return {
            'total_agents': len(self.controlled_keys),
            'total_trials': self.k_trials,
            'history_steps': self.vectorized_history.shape[2],
            'feature_size': self.vectorized_history.shape[3],
            'non_zero_entries': non_zero_entries.item(),
            'total_entries': total_entries,
            'sparsity': 1.0 - (non_zero_entries.float() / total_entries).item(),
            'memory_usage_mb': memory_usage
        }

