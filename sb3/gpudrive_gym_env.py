"""Gym Environment that interfaces with the GPU Drive simulator."""

from dataclasses import dataclass
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import islice, product
# Simulator
import gpudrive
import logging

logging.getLogger(__name__)

# Utils
from logger import MetricsLogger

class Env:  
    """
    GPU Drive Gym Environment.
    """
    def __init__(self, num_worlds, max_cont_agents, data_dir="nocturne_data", device="cuda", auto_reset=True):
        
        # Configure rewards 
        #TODO: Make this configurable on the Python side and add docs
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = gpudrive.RewardType.DistanceBased  # Or any other value from the enum
        reward_params.distanceToGoalThreshold = 1.0  # Set appropriate values
        reward_params.distanceToExpertThreshold = 1.0  # Set appropriate values
        
        # Configure the environment
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = 0.5 
        params.observationRadius = 10.0  
        params.rewardParams = reward_params 
        
        # Set number of controlled vehicles
        params.maxNumControlledVehicles = max_cont_agents
    
        self.data_dir = data_dir
        self.num_sims = num_worlds
        self.device = device    
        self.action_types = 3
        
        # Initialize the simulator
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CPU if device == 'cpu' else gpudrive.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=self.num_sims,
            auto_reset=auto_reset,
            json_path=self.data_dir,
            params=params,
        ) 
        
        # We only want to obtain information from vehicles we control
        # By default, the sim returns information for all vehicles in a scene 
        # We construct a mask to filter out the information from the expert controlled vehicles (0)
        #TODO: Check
        self.all_agents = self.sim.controlled_state_tensor().to_torch().shape[1]
        self.cont_agent_idx = torch.where(self.sim.controlled_state_tensor().to_torch()[0, :, 0] == 1)[0]
        self.max_cont_agents = max_cont_agents
        
        # Set up action space (TODO: Make this work for multiple worlds)
        self.action_space = self._set_discrete_action_space()
        
        # Set observation space        
        self.observation_space = self._set_observation_space()

        # Configure logger
        # TODO: Complete logger
        self.metrics = MetricsLogger({"num_sims": self.num_sims, "num_agents": self.max_cont_agents, "device": device})
                
    def reset(self):
        """Reset the worlds and return the initial observations."""
        for sim_idx in range(self.num_sims):
            self.sim.reset(sim_idx)
            
        obs = self.get_obs()
        # Filter out the observations for the expert controlled vehicles, 
        # Make sure shape is consistent: (num_worlds, max_cont_agents, num_features)
        obs = torch.index_select(obs, dim=1, index=self.cont_agent_idx).reshape(self.num_sims, self.max_cont_agents, -1)
        return obs

    def step(self, actions):
        """Take simultaneous actions for each agent in all `num_worlds` environments.

        Args:
            actions (torch.Tensor): The action indices for all agents in all worlds.
        """
        # Convert action indices to action values
        actions_shape = self.sim.action_tensor().to_torch().shape[1]
        action_values = torch.zeros((self.num_sims, actions_shape, self.action_types))
        
        # GPU Drive expects a tensor of shape (num_worlds, all_agents)
        # We insert the actions for the controlled vehicles, the others will be ignored
        for agent_idx in range(self.cont_agent_idx.shape[0]):
            action_idx = actions[:, agent_idx].item()
            action_values[:, agent_idx, :] = torch.Tensor(self.action_key_to_values[action_idx])

        # Feed the actual action values to GPU Drive 
        self.sim.action_tensor().to_torch().copy_(action_values)
        
        # Step the simulator
        self.sim.step()
        
        # Obtain the next observations, rewards, and done flags
        obs = self.get_obs()
        reward = self.sim.reward_tensor().to_torch()
        done = self.sim.done_tensor().to_torch()
        
        # Filter out the expert controlled vehicle information
        obs = torch.index_select(obs, dim=1, index=self.cont_agent_idx).reshape(self.num_sims, self.max_cont_agents, -1)
        reward = torch.index_select(reward, dim=1, index=self.cont_agent_idx).reshape(self.num_sims, self.max_cont_agents, -1)
        done = torch.index_select(done, dim=1, index=self.cont_agent_idx).reshape(self.num_sims, self.max_cont_agents, -1)
        
        # The episode is reset automatically if all agents reach their goal
        # or the episode is over 
        # We also reset when all agents have collided
        is_collided = torch.index_select(self.sim.self_observation_tensor().to_torch()[:, :, 5], dim=1, index=self.cont_agent_idx)
        
        return obs, reward, done, {}
    
    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space."""
        
        self.steer_actions = torch.tensor([-0.6, 0, 0.6], device=self.device)
        self.accel_actions = torch.tensor([-3, 0, 3], device=self.device)
        self.head_actions = torch.tensor([0], device=self.device)

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
    
        for action_idx, (accel, steer, head) in  enumerate(product(self.accel_actions, self.steer_actions, self.head_actions)):  
            self.action_key_to_values[action_idx] = [int(accel), int(steer), int(head)]   
    
        return Discrete(n=int(len(self.action_key_to_values)))
    
    def _set_observation_space(self) -> None:
        """Configure the observation space."""
        return Box(low=-np.inf, high=np.inf, shape=(self.get_obs().shape[-1],))
    
    def get_obs(self):
        """Get observation: Combine different types of environment information into a single tensor.
        
        Returns:
            torch.Tensor: (num_worlds, max_cont_agents, num_features)
        """       
        # Get the ego states
        # Ego state: (num_worlds, max_cont_agents, features)
        ego_state = self.sim.self_observation_tensor().to_torch()
        
        # Get view of other agents
        # Partner obs: (num_worlds, max_cont_agents, max_cont_agents-1, num_features)
        # Flatten over the last two dimensions to get (num_worlds, max_cont_agents, (max_cont_agents-1) * num_features)
        partner_obs_tensor = self.sim.partner_observations_tensor().to_torch().flatten(start_dim=2)
        
        # Get view of road graphs
        # Roadmap obs: (num_worlds, max_cont_agents, max_road_points, num_features)
        # Flatten over the last two dimensions to get (num_worlds, max_cont_agents, max_road_points * num_features)
        map_obs_tensor = self.sim.agent_roadmap_tensor().to_torch().flatten(start_dim=2)
        
        return torch.cat([ego_state, partner_obs_tensor, map_obs_tensor], dim=-1)
    

if __name__ == "__main__":
    
    # Using a single world with 13 agents, controlling only a single one
    env = Env(num_worlds=1, max_cont_agents=2, device='cuda')
    obs = env.reset()
    
    # actions.shape: (num_worlds, max_cont_agents)
    
    rand_actions = torch.ones((env.num_sims, env.max_cont_agents))
    obs, reward, done, info = env.step(rand_actions)
    
    # obs.shape: (num_worlds, max_cont_agents, 20699)
    # reward.shape: (num_worlds, max_cont_agents, 1)
    # done.shape: (num_worlds, max_cont_agents, 1)