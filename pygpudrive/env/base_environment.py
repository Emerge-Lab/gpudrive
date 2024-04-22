"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import product

# import wandb
import glob
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import pygame
import random
import os
import math

# Import the EnvConfig dataclass
from pygpudrive.env.config import EnvConfig

# Import the simulator
import gpudrive
import logging

logging.getLogger(__name__)

WINDOW_W = 500
WINDOW_H = 500
WINDOW_SIZE = (WINDOW_W, WINDOW_H)
VEH_WIDTH = 5
VEH_HEIGHT = 10
GOAL_RADIUS = 2
COLOR_LIST = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
]

# https://stackoverflow.com/a/73855696
def compute_agent_corners(center, width, height, rotation):
    """Draw a rectangle, centered at x, y.

    Arguments:
      x (int/float):
        The x coordinate of the center of the shape.
      y (int/float):
        The y coordinate of the center of the shape.
      width (int/float):
        The width of the rectangle.
      height (int/float):
        The height of the rectangle.
    """
    x, y = center

    points = []

    # The distance from the center of the rectangle to
    # one of the corners is the same for each corner.
    radius = math.sqrt((height / 2) ** 2 + (width / 2) ** 2)

    # Get the angle to one of the corners with respect
    # to the x-axis.
    angle = math.atan2(height / 2, width / 2)

    # Transform that angle to reach each corner of the rectangle.
    angles = [angle, -angle + math.pi, angle + math.pi, -angle]

    # Calculate the coordinates of each point.
    for angle in angles:
        y_offset = -1 * radius * math.sin(angle + rotation)
        x_offset = radius * math.cos(angle + rotation)
        points.append((x + x_offset, y + y_offset))

    return points


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
        auto_reset=False,
        render_mode="rgb_array",
        verbose=True,
    ):
        self.config = config

        # Configure rewards
        # TODO: Make this configurable on the Python side and add docs
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = (
            gpudrive.RewardType.OnGoalAchieved
        )  # Or any other value from the enum
        reward_params.distanceToGoalThreshold = (
            self.config.dist_to_goal_threshold  # Set appropriate values
        )
        reward_params.distanceToExpertThreshold = 1.0  # Set appropriate values

        # Configure the environment
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = 0.5
        params.observationRadius = 10.0
        params.rewardParams = reward_params

        # Collision behavior
        if self.config.collision_behavior == "ignore":
            params.collisionBehaviour = gpudrive.CollisionBehaviour.Ignore
        elif self.config.collision_behavior == "remove":
            params.collisionBehaviour = (
                gpudrive.CollisionBehaviour.AgentRemoved
            )
        elif self.config.collision_behavior == "stop":
            params.collisionBehaviour = gpudrive.CollisionBehaviour.AgentStop
        else:
            raise ValueError(
                f"Invalid collision behavior: {self.config.collision_behavior}"
            )

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
        # Get the indices of the controlled agents
        self.w_indices, self.k_indices = torch.where(self.cont_agent_mask)
        self.max_cont_agents = max_cont_agents

        # Number of valid controlled agents (without padding agents)
        self.num_valid_controlled_agents = self.cont_agent_mask.sum().item()

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

        # Convert nan values to 0 for dead agents
        # This does not (these actions will be ignored)
        actions = torch.nan_to_num(actions, nan=0)

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
            torch.empty(self.num_sims, self.max_cont_agents)
            .fill_(float("nan"))
            .to(self.device)
        )
        reward_full = self.sim.reward_tensor().to_torch().squeeze(dim=2)
        reward[self.w_indices, self.k_indices] = reward_full[
            self.w_indices, self.k_indices
        ]

        # Return 1s for invalid (padding) agents
        done = (
            torch.empty(self.num_sims, self.max_cont_agents)
            .fill_(1)
            .to(self.device)
        )
        done_full = (
            self.sim.done_tensor().to_torch().squeeze(dim=2).to(done.dtype)
        )
        done[self.w_indices, self.k_indices] = done_full[
            self.w_indices, self.k_indices
        ]

        # TODO: add info
        info = torch.zeros_like(done)

        return obs, reward, done, info

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space."""

        self.steer_actions = self.config.steer_actions.to(self.device)
        self.accel_actions = self.config.accel_actions.to(self.device)
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

        # Get the ego state
        # Ego state: (num_worlds, kMaxAgentCount, features)
        if self.config.ego_state:
            ego_state_padding = (
                torch.empty(self.num_sims, self.max_cont_agents, 6)
                .fill_(float("nan"))
                .to(self.device)
            )
            full_ego_state = self.sim.self_observation_tensor().to_torch()

            # Update ego_state_padding using the mask
            ego_state_padding[
                self.w_indices, self.k_indices, :
            ] = full_ego_state[self.w_indices, self.k_indices, :]

            # Normalize
            if self.config.normalize_obs:
                ego_state_padding = self.normalize_ego_state(ego_state_padding)
        else:
            ego_state_padding = torch.Tensor().to(self.device)

        # Get patner observation
        # Partner obs: (num_worlds, kMaxAgentCount, kMaxAgentCount - 1 * num_features)
        if self.config.partner_obs:
            partner_obs_padding = (
                torch.empty(self.num_sims, self.max_cont_agents, 4)
                .fill_(float("nan"))
                .to(self.device)
            )
            full_partner_obs = (
                self.sim.partner_observations_tensor()
                .to_torch()
                .flatten(start_dim=2)
            )

            partner_obs_padding[
                self.w_indices, self.k_indices, :
            ] = full_partner_obs[self.w_indices, self.k_indices, :]

            # Normalize
            if self.config.normalize_obs:
                partner_obs_padding = self.normalize_partner_obs(
                    partner_obs_padding
                )
        else:
            partner_obs_padding = torch.Tensor().to(self.device)

        # Get road map
        # Roadmap obs: (num_worlds, kMaxAgentCount, kMaxRoadEntityCount, num_features)
        # Flatten over the last two dimensions to get (num_worlds, kMaxAgentCount, kMaxRoadEntityCount * num_features)
        if self.config.road_map_obs:
            map_obs_padding = (
                torch.empty(self.num_sims, self.max_cont_agents, 4)
                .fill_(float("nan"))
                .to(self.device)
            )
            full_map_obs = (
                self.sim.agent_roadmap_tensor().to_torch().flatten(start_dim=2)
            )

            map_obs_padding[self.w_indices, self.k_indices, :] = full_map_obs[
                self.w_indices, self.k_indices, :
            ]

            # TODO: Normalize
        else:
            map_obs_padding = torch.Tensor().to(self.device)

        # Combine the observations
        obs_filtered = torch.cat(
            (
                ego_state_padding,
                partner_obs_padding,
                map_obs_padding,
            ),
            dim=-1,
        )

        return obs_filtered

    def render(self):
        """Render the environment."""

        def create_render_mask():
            agent_to_is_valid = (
                self.sim.valid_state_tensor()
                .to_torch()[self.world_render_idx, :, :]
                .cpu()
                .detach()
                .numpy()
            )

            return agent_to_is_valid

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

        # Get agent info
        agent_info = (
            self.sim.absolute_self_observation_tensor()
            .to_torch()[self.world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )

        # Get the agent goal positions and current positions
        agent_pos = agent_info[:, :2]  # x, y
        agent_rot = agent_info[:, 7]
        goal_pos = agent_info[:, 8:]

        render_mask = create_render_mask()

        agent_pos_mean = np.mean(agent_pos, axis=0, where=render_mask == 1)

        num_agents_in_scene = np.count_nonzero(goal_pos[:, 0])

        # Draw the agent positions
        for agent_idx in range(num_agents_in_scene):
            if not render_mask[agent_idx]:
                continue

            # Use the updated scale_coord function to get centered and scaled coordinates
            current_pos_scaled = self.scale_coords(
                agent_pos[agent_idx], agent_pos_mean[0], agent_pos_mean[1]
            )

            current_goal_scaled = self.scale_coords(
                goal_pos[agent_idx], agent_pos_mean[0], agent_pos_mean[1]
            )

            mod_idx = agent_idx % len(COLOR_LIST)

            agent_corners = compute_agent_corners(
                current_pos_scaled, VEH_WIDTH, VEH_HEIGHT, agent_rot[agent_idx]
            )

            pygame.draw.polygon(
                surface=self.surf,
                color=COLOR_LIST[mod_idx],
                points=agent_corners,
            )

            pygame.draw.circle(
                surface=self.surf,
                color=COLOR_LIST[mod_idx],
                center=(
                    int(current_goal_scaled[0]),
                    int(current_goal_scaled[1]),
                ),
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

    def scale_coords(self, coords, x_avg, y_avg):
        """Scale the coordinates to fit within the pygame surface window and center them.
        Args:
            coords: x, y coordinates
        """
        x, y = coords

        x_scaled = x - x_avg + (WINDOW_W / 2)
        y_scaled = y - y_avg + (WINDOW_H / 2)

        return (x_scaled, y_scaled)

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= self.config.max_speed
        state[:, :, 1] /= self.config.max_veh_len
        state[:, :, 2] /= self.config.max_veh_width

        # Relative goal coordinates
        state[:, :, 3] /= self.config.max_rel_goal_coord
        state[:, :, 4] /= self.config.max_rel_goal_coord

        return state

    def normalize_partner_obs(self, state):
        """Normalize ego state features."""

        # print(state.shape)
        # print(f'max_before: {state.max()}')
        # Speed, vehicle length, vehicle width
        state /= self.config.max_partner

        # print(f'max_after: {state.max()}')

        return state

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
    # run = wandb.init(
    #     project="gpudrive",
    #     group="test_rendering",
    # )
    NUM_CONT_AGENTS = 10
    NUM_WORLDS = 1

    env = Env(
        config=config,
        num_worlds=NUM_WORLDS,
        max_cont_agents=NUM_CONT_AGENTS,
        data_dir="waymo_data",
        device="cpu",
    )

    obs = env.reset()
    frames = []

    for _ in range(100):

        print(f"Step: {90 - env.steps_remaining[0, 0, 0].item()}")

        # Take a random action (we're only going straight)
        rand_action = torch.Tensor(
            [
                [
                    env.action_space.sample()
                    for _ in range(NUM_CONT_AGENTS * NUM_WORLDS)
                ]
            ]
        ).reshape(NUM_WORLDS, NUM_CONT_AGENTS)

        # Step the environment
        obs, reward, done, info = env.step(rand_action)

        print(obs.max())

        if done.sum() == NUM_CONT_AGENTS:
            obs = env.reset()
            print(f"RESETTING ENVIRONMENT\n")

        frame = env.render()
        frames.append(frame.T)

    # Log video
    # wandb.log({"scene": wandb.Video(np.array(frames), fps=10, format="gif")})
    # wandb.log({"scene": wandb.Video(np.array(frames), fps=10, format="gif")})

    # run.finish()
    env.close()
