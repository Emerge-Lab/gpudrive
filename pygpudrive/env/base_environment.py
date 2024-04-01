"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import product

import glob
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import pygame
import random
import os

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig


# Import the simulator
import gpudrive
import logging

logging.getLogger(__name__)

WINDOW_W = 500
WINDOW_H = 500
VEH_WIDTH = 25
VEH_HEIGHT = 50
GOAL_RADIUS = 10
COLOR_LIST = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
]


class Env(gym.Env):
    """
    GPU Drive Gym Environment.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],  # human: pop-up, rgb_array: receive the visualization as an array of pixels
        "render_fps": 5,
    }

    def __init__(
        self,
        config,
        num_worlds,
        max_cont_agents,
        data_dir,
        device="cuda",
        auto_reset=True,
        render_mode="rgb_array",
        verbose=True,
    ):
        self.config = config

        # Configure rewards
        # TODO: Make this configurable on the Python side and add docs
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = (
            gpudrive.RewardType.DistanceBased
        )  # Or any other value from the enum
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
        self.action_types = 3  # Acceleration, steering, and heading

        # Rendering
        self.render_mode = render_mode
        self.world_render_idx = 0  # Render only the 0th world
        self.screen_dim = 1000
        self.screen = None
        self.clock = None

        # TODO: @AP / @SK: Allow for num_worlds < num_files
        assert num_worlds == len(
            glob.glob(f"{data_dir}/*.json")
        ), "Number of worlds exceeds the number of files in the data directory."

        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            assert False, "The data directory does not exist or is empty."

        # Initialize the simulator
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CPU
            if device == "cpu"
            else gpudrive.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=self.num_sims,
            auto_reset=auto_reset,
            json_path=self.data_dir,
            params=params,
        )

        # We only want to obtain information from vehicles we control
        # By default, the sim returns information for all vehicles in a scene
        # We construct a mask to filter out the information from the expert controlled vehicles (0)
        # Mask size: (num_worlds, kMaxAgentCount)
        self.cont_agent_mask = (
            self.sim.controlled_state_tensor().to_torch() == 1
        ).squeeze(dim=2)
        self.max_cont_agents = max_cont_agents

        # Set up action space
        self.action_space = self._set_discrete_action_space()

        # Set observation space
        self.observation_space = self._set_observation_space()
        self.obs_dim = self.observation_space.shape[0]

        self._print_info(verbose)

    def reset(self):
        """Reset the worlds and return the initial observations."""
        for sim_idx in range(self.num_sims):
            self.sim.reset(sim_idx)

        return self.get_obs()

    def step(self, actions):
        """Take simultaneous actions for each agent in all `num_worlds` environments.

        Args:
            actions (torch.Tensor): The action indices for all agents in all worlds.
        """
        assert actions.shape == (
            self.num_sims,
            self.max_cont_agents,
        ), """Action tensor must match the shape (num_worlds, max_cont_agents)"""

        # Convert action indices to action values
        actions_shape = self.sim.action_tensor().to_torch().shape[1]
        action_value_tensor = torch.zeros(
            (self.num_sims, actions_shape, self.action_types)
        )

        # GPU Drive expects a tensor of shape (num_worlds, kMaxAgentCount, 3)
        # We insert the actions for the controlled vehicles, the others will be ignored
        for world_idx in range(self.num_sims):
            for agent_idx in range(self.max_cont_agents):
                action_idx = actions[world_idx, agent_idx].item()
                action_value_tensor[world_idx, agent_idx, :] = torch.Tensor(
                    self.action_key_to_values[action_idx]
                )

        # Feed the actual action values to GPU Drive
        self.sim.action_tensor().to_torch().copy_(action_value_tensor)

        # Step the simulator
        self.sim.step()

        # Obtain the next observations, rewards, and done flags
        obs = self.get_obs()
        reward = (
            self.sim.reward_tensor().to_torch()[self.cont_agent_mask]
        ).reshape(self.num_sims, self.max_cont_agents)
        done = (
            self.sim.done_tensor().to_torch()[self.cont_agent_mask]
        ).reshape(self.num_sims, self.max_cont_agents)

        # TODO: add info
        info = torch.zeros_like(done)

        # The episode is reset automatically if all agents reach their goal
        # or the episode is over
        # TODO: also reset when all agents have collided

        return obs, reward, done, info

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space."""

        self.steer_actions = torch.tensor([-0.6, 0, 0.6], device=self.device)
        self.accel_actions = torch.tensor([-3, 0, 3], device=self.device)
        self.head_actions = torch.tensor([0], device=self.device)

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}

        for action_idx, (accel, steer, head) in enumerate(
            product(self.accel_actions, self.steer_actions, self.head_actions)
        ):
            self.action_key_to_values[action_idx] = [
                accel.item(),
                steer.item(),
                head.item(),
            ]

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
        # Ego state: (num_worlds, kMaxAgentCount, features)
        if self.config.ego_state:
            ego_state = self.sim.self_observation_tensor().to_torch()
        else:
            ego_state = torch.Tensor().to(self.device)

        # Get patner obs view
        # Partner obs: (num_worlds, kMaxAgentCount, kMaxAgentCount - 1 * num_features)
        if self.config.partner_obs:
            partner_obs_tensor = (
                self.sim.partner_observations_tensor()
                .to_torch()
                .flatten(start_dim=2)
            )
        else:
            partner_obs_tensor = torch.Tensor().to(self.device)

        # Get road map
        # Roadmap obs: (num_worlds, kMaxAgentCount, kMaxRoadEntityCount, num_features)
        # Flatten over the last two dimensions to get (num_worlds, kMaxAgentCount, kMaxRoadEntityCount * num_features)
        if self.config.road_map_obs:
            map_obs_tensor = (
                self.sim.agent_roadmap_tensor().to_torch().flatten(start_dim=2)
            )
        else:
            map_obs_tensor = torch.Tensor().to(self.device)

        # Combine the observations
        obs_all = torch.cat(
            (ego_state, partner_obs_tensor, map_obs_tensor), dim=-1
        )

        # Only select the observations for the controlled agents
        obs_filtered = obs_all[self.cont_agent_mask].reshape(
            self.num_sims, self.max_cont_agents, -1
        )

        return obs_filtered

    def render(self, t=None):
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
        self.surf.fill((255, 255, 255))  # White background

        # # # Draw the road map # # #
        roadmap_info = (
            self.sim.map_observation_tensor()
            .to_torch()[self.world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )
        roadmap_pos = roadmap_info[:, :2]
        roadmap_heading = roadmap_info[:, 2:]
        scaled_rm_positions = [
            self.scale_coord(pos, roadmap_pos) for pos in roadmap_pos
        ]
        pygame.draw.lines(self.surf, (0, 0, 0), False, scaled_rm_positions)

        # Draw the agents
        # We only render agents from the chosen world index
        # Access the positions of the agents
        agent_info = (
            self.sim.absolute_self_observation_tensor()
            .to_torch()[self.world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )
        agent_positions = agent_info[:, :3]  # x, y, z
        agent_rot_quaternions = agent_info[:, 3:7]  # rotation as quaternion
        agent_rot_rad = agent_info[:, 7:8]  # rotation from x-axis in radians
        agent_goal_positions = agent_info[:, 8:]

        # Dynamically adjust the lower and upper bounds of the frame
        frame_xy_coords = np.concatenate(
            [agent_goal_positions, agent_positions[:, :2]]
        )

        # Draw the agent positions and goal positions with adjustments
        for agent_idx in range(agent_info.shape[0]):

            # Scale positions to fit within window
            current_pos_screen = self.scale_coord(
                agent_positions[agent_idx], frame_xy_coords
            )
            goal_pos_screen = self.scale_coord(
                agent_goal_positions[agent_idx], frame_xy_coords
            )

            # Randomly sample a color from the color list for the agent
            agent_color = random.choice(COLOR_LIST)

            # Draw the current agent position with the randomly chosen color
            pygame.draw.rect(
                surface=self.surf,
                color=agent_color,  # Use the randomly sampled color
                rect=pygame.Rect(
                    int(current_pos_screen[0]),
                    int(current_pos_screen[1]),
                    VEH_WIDTH,
                    VEH_HEIGHT,
                ),
            )

            # Draw the goal position
            pygame.draw.circle(
                surface=self.surf,
                color=(0, 255, 0),  # Green
                center=(int(goal_pos_screen[0]), int(goal_pos_screen[1])),
                radius=GOAL_RADIUS,
            )

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._create_image_array(self.surf)
        else:
            return self.isopen

    def scale_coord(self, pos, frame_xy_coords):
        """Scale the coordinates to fit within the pygame surface window."""

        # Extract the lower and upper bounds of the frame
        x_min, x_max = frame_xy_coords[:, 0].min(), frame_xy_coords[:, 0].max()
        y_min, y_max = frame_xy_coords[:, 1].min(), frame_xy_coords[:, 1].max()

        # Scale coordinates
        x_scaled = ((pos[0] - x_min) / (x_max - x_min)) * WINDOW_W
        y_scaled = ((pos[1] - y_min) / (y_max - y_min)) * WINDOW_H
        return [x_scaled, y_scaled]

    def close(self):
        """Close pygame application if open."""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _create_image_array(self, surf):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )

    def _print_info(self, verbose=True):
        """Print initialization information."""
        if verbose:
            logging.info("----------------------")
            logging.info(f"Device: {self.device}")
            logging.info(f"Number of worlds: {self.num_sims}")
            logging.info(
                f"Number of maps in data directory: {len(glob.glob(f'{self.data_dir}/*.json'))}"
            )
            logging.info(
                f"Number of controlled agents: {self.max_cont_agents}"
            )
            logging.info("----------------------\n")

    @property
    def steps_remaining(self):
        return self.sim.steps_remaining_tensor().to_torch()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    config = EnvConfig()

    env = Env(
        config=config,
        num_worlds=1,
        max_cont_agents=1,  # Number of agents to control
        data_dir="waymo_data",
        device="cuda",
    )

    obs = env.reset()

    for _ in range(200):

        print(
            f"Remaining steps in episode: {env.steps_remaining[0, 0, 0].item()}"
        )

        # Take a random action
        rand_action = torch.Tensor([[env.action_space.sample()]])

        # Step the environment
        obs, reward, done, info = env.step(rand_action)

        print(
            f"action (acc, steer, heading): {env.action_key_to_values[rand_action.item()]} | reward: {reward.item():.3f}"
        )
        print(
            f"veh_x_pos: {obs[:, :, 3].item():.3f} | veh_y_pos: {obs[:, :, 4].item():.3f}"
        )
        print(f"veh_speed: {obs[:, :, 0].item():.3f}")
        print(f"done: {done}")

        print(f"veh_collided: {obs[:, :, 5]}")

        print(f"--- \n")
        # frame = env.render(i)
        # frames.append(frame.T)

        if done.all():
            obs = env.reset()

    env.close()
