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

# Import the simulator
import gpudrive
import logging

logging.getLogger(__name__)

VIDEO_W = 4000
VIDEO_H = 4000
WINDOW_W = 1000
WINDOW_H = 800
ZOOM = 2.7  # Camera zoom
SCALE = 30

class Env(gym.Env):  
    """
    GPU Drive Gym Environment.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"], # human: pop-up, rgb_array: receive the visualization as an array of pixels
        "render_fps": 5,
    }

    def __init__(self, num_worlds, max_cont_agents, data_dir, device="cuda", auto_reset=True, render_mode="rgb_array",):
        
        # Configure rewards 
        #TODO: Make this configurable on the Python side and add docs
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
        self.screen_dim = 1000
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

        pygame.init()
        pygame.font.init()
        
        if self.screen is None and self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Create a new canvas to draw on    
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))  
        self.surf.fill((255, 255, 0)) # Yellow background test
        
        # Draw the agents
        # We only render agents from the chosen world index
        # Access the positions of the agents
        agent_info = self.sim.absolute_self_observation_tensor().to_torch()[
            self.world_render_idx, :, :].cpu().detach().numpy()
        agent_positions = agent_info[:, :3] # x, y, z
        agent_rot_quaternions = agent_info[:, 3:7] # rotation as quaternion
        agent_rot_rad = agent_info[:, 7:8] # rotation from x-axis in radians
        agent_goal_positions = agent_info[:, 8:]
        
         # New scale factors for drawing, assuming the simulation coordinates can range widely
        scale_factor_x = WINDOW_W / (VIDEO_W * 2)  # Adjust based on your simulation's coordinate range
        scale_factor_y = WINDOW_H / (VIDEO_H * 2)  # Adjust similarly

        # Draw the agent positions and goal positions with adjustments
        for agent_idx in range(agent_info.shape[0]):
            current_pos = agent_positions[agent_idx]
            goal_pos = agent_goal_positions[agent_idx]

            # Scale and translate positions to fit within window
            current_pos_screen = (int((current_pos[0] - VIDEO_W / 2) * scale_factor_x + WINDOW_W / 2),
                                int((current_pos[1] - VIDEO_H / 2) * scale_factor_y + WINDOW_H / 2))
            goal_pos_screen = (int((goal_pos[0] - VIDEO_W / 2) * scale_factor_x + WINDOW_W / 2),
                            int((goal_pos[1] - VIDEO_H / 2) * scale_factor_y + WINDOW_H / 2))

            # Adjust rectangle size for visibility
            agent_rect_size = (10, 10)  # Adjust as needed for visibility

            # Draw the agent
            pygame.draw.rect(
                self.surf,
                (255, 0, 0),  # Red color for the agent
                pygame.Rect(
                    current_pos_screen[0] - agent_rect_size[0] / 2,
                    current_pos_screen[1] - agent_rect_size[1] / 2,
                    agent_rect_size[0],
                    agent_rect_size[1],
                ),
            )

        # Draw the goal position with a smaller, visible circle
        pygame.draw.circle(
            surface=self.surf,
            color=(0, 255, 0),  # Green color for the goal
            center=goal_pos_screen,
            radius=5,  # Smaller radius for visibility
        )

        # TODO: Draw the road graph       
        #self._render_roadmap(zoom)
        # pygame.draw.line(self.screen, (60, 179, 113), [0, 0], [50, 30], 5)
    
        # # Draw a circle
        # pygame.draw.circle(self.screen, "blue", [60, 250], 40)
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        else:
            return self.isopen
            
    def close(self):
        """Close pygame application if open."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.scale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

if __name__ == "__main__":
    
    run = wandb.init(
        project="gpu_drive", 
        group='test_rendering',
    )

    env = Env(
        num_worlds=1, 
        max_cont_agents=1, 
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
        
        frames.append(frame.T)

    # Log video
    wandb.log({"scene": wandb.Video(np.array(frames), fps=4, format="gif")})
    
    env.close()