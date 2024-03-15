"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from dataclasses import dataclass
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import islice, product
import wandb

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import pygame
from pygame import gfxdraw

# Simulator
import gpudrive
import logging

logging.getLogger(__name__)

class Env(gym.Env):  
    """
    GPU Drive Gym Environment.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"], # human: pop-up, rgb_array: receive the visualization as an array of pixels
        "render_fps": 5,
    }

    def __init__(self, num_worlds, max_cont_agents, data_dir, render_mode="rgb_array", device="cuda", auto_reset=True):
        
        # Configure rewards 
        #TODO: Make this configurable through dataclasses on the Python side and add docs
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = gpudrive.RewardType.DistanceBased  # Or any other value from the enum
        reward_params.distanceToGoalThreshold = 2.0  # Set appropriate values
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
        self.action_types = 3 # Acceleration, steering, and heading
        
        # Rendering
        self.render_mode = render_mode
        self.world_render_idx = 0 # Render only the 0th world
        self.screen_dim = 800
        self.screen = None
        self.clock = None
        
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
        #TODO: Get this to work for multiple worlds
        self.all_agents = self.sim.controlled_state_tensor().to_torch().shape[1]
        self.cont_agent_idx = torch.where(self.sim.controlled_state_tensor().to_torch()[0, :, 0] == 1)[0]
        self.max_cont_agents = max_cont_agents
        
        # Set up action space (TODO: Make this work for multiple worlds)
        self.action_space = self._set_discrete_action_space()
        
        # Set observation space        
        self.observation_space = self._set_observation_space()
                
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
        # TODO: add info
        info = torch.zeros_like(done)
        
        # The episode is reset automatically if all agents reach their goal
        # or the episode is over 
        #TODO: also reset when all agents have collided
        is_collided = torch.index_select(self.sim.self_observation_tensor().to_torch()[:, :, 5], dim=1, index=self.cont_agent_idx)
        
        return obs, reward, done, info
    
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
    
    
    def render(self):
        """Render the environment."""
        
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        try:
            import pygame
            
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else: # mode in "rgb_array"
                #self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
                self.screen = pygame.Surface((400, 300))
                self.screen.blit(self.surf, (0, 0))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
       
        # Create a background surface
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((0, 255, 255)) # Fill the surface with white
        
        # Draw a circle
        pygame.draw.circle(self.surf, "blue", [60, 250], 40)
       
        #TODO: Draw the road graph
        #obj = obj.cpu().numpy()
        # pygame.draw.rect(
        #     surface=self.surf,
        #     color=(255, 0, 0),
        #     rect=pygame.Rect(0, 5, 100, 50),  # x, y, width, height
        #     width=2,
        # )
        
        # # Draw the agents
        # # We only render agents from the chosen world index
        # # Access the positions of the agents
        # agent_info = self.sim.absolute_self_observation_tensor().to_torch()[
        #     self.world_render_idx, :, :].cpu().detach().numpy()
        # agent_positions = agent_info[:, :3] # x, y, z
        # agent_rot_quaternions = agent_info[:, 3:7] # rotation as quaternion
        # agent_rot_rad = agent_info[:, 7:8] # rotation from x-axis in radians
        # agent_goal_positions = agent_info[:, 8:]
        
        # # Draw the agent positions and goal positions
        # for agent_idx in range(agent_info.shape[0]):
            
        #     current_pos = agent_positions[agent_idx]
        #     goal_pos = agent_goal_positions[agent_idx]
            
        #     print(f'agent_idx: {agent_idx}, current_pos: {current_pos}, goal_pos: {goal_pos}')
            
        #     # Draw the agent
        #     pygame.draw.rect(
        #         surface=self.screen,
        #         color=(255, 0, 0),
        #         rect=pygame.Rect(int(current_pos[0]), int(current_pos[0]), 100, 50), # x, y, width, height
        #         width=2,
        #     )
            
        #     # Draw the goal position
        #     pygame.draw.circle(
        #         surface=self.screen,
        #         color=(255, 0, 0),
        #         center=(
        #             int(goal_pos[0]),
        #             int(goal_pos[1]),
        #         ),
        #         radius=200,
        #     )
        
        # # Draw the agent goals
        # for i, goal in enumerate(self.goals[0]):
        #     goal = goal.cpu().numpy()
        #     x = int(goal[0] + self.cfg.warehouse_width)
        #     y = int(goal[1] + self.cfg.warehouse_width)
        #     pygame.draw.circle(
        #         self.screen,
        #         (0, 0, 255),
        #         (int(goal[0] + self.cfg.warehouse_width), int(goal[1] + self.cfg.warehouse_width)),
        #         self.cfg.goal_radius,
        #     )

        #     # Render the number in red. True stands for anti-aliasing.
        #     text_surface = font.render(str(i), True, (255, 0, 0))

        #     # The x, y position where you want to draw the number
        #     # Draw the text surface at the x, y position
        #     self.screen.blit(text_surface, (x, y))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array": return the rendered image
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        """Close pygame application if open."""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

if __name__ == "__main__":
    
    run = wandb.init(
        project="gpu_drive", 
        group='test_rendering',
    )

    env = Env(
        num_worlds=1, 
        max_cont_agents=1, 
        render_mode='rgb_array', 
        data_dir='waymo_data', 
        device='cuda'
    )
    
    obs = env.reset()
    
    # actions.shape: (num_worlds, max_cont_agents)
    rand_actions = torch.ones((env.num_sims, env.max_cont_agents))
    obs, reward, done, info = env.step(rand_actions)
                
    # obs.shape: (num_worlds, max_cont_agents, 20699)
    # reward.shape: (num_worlds, max_cont_agents, 1)
    # done.shape: (num_worlds, max_cont_agents, 1)
    
    frames = []
    
    for i in range(5):
        print(f"Step {i}")
        obs, reward, done, info = env.step(rand_actions)
        frame = env.render()
        
        frames.append(frame)

    # Log video
    wandb.log({"scene": wandb.Video(np.array(frames), fps=4, format="gif")})
    
    env.close()