"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import product
import wandb
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
        reward_params.distanceToGoalThreshold = 3.0  # Set appropriate values
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
        ), "Number of worlds is not equal to the number of files in the data directory."

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
        # Get the ego states
        # Ego state: (num_worlds, kMaxAgentCount, features)
        if self.config.ego_state:
            ego_state = self.sim.self_observation_tensor().to_torch()

            # Filter out the information for the controlled agents
            ego_state = ego_state[self.cont_agent_mask].reshape(
                self.num_sims, self.max_cont_agents, -1
            )

            if not self.config.collision_state:
                # Remove collision state from SelfObservation
                ego_state = ego_state[:, :, :-3]

            if self.config.normalize_obs:  # Normalize
                ego_state = self.normalize_ego_state(ego_state)
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
            # TODO: Normalize

        else:
            partner_obs_tensor = torch.Tensor().to(self.device)

        # Get road map
        # Roadmap obs: (num_worlds, kMaxAgentCount, kMaxRoadEntityCount, num_features)
        # Flatten over the last two dimensions to get (num_worlds, kMaxAgentCount, kMaxRoadEntityCount * num_features)
        if self.config.road_map_obs:
            map_obs_tensor = (
                self.sim.agent_roadmap_tensor().to_torch().flatten(start_dim=2)
            )
            # TODO: Normalize
        else:
            map_obs_tensor = torch.Tensor().to(self.device)

        # Get agent info
        agent_info = self.sim.absolute_self_observation_tensor().to_torch()
        # Get the agent goal positions and current positions
        goal_pos = agent_info[:, :, 8:]
        agent_pos = agent_info[:, :, :2]  # x, y

        # L2 norm to goal position
        if self.config.goal_dist:
            goal_dist_tensor = torch.linalg.norm(
                agent_pos - goal_pos, dim=-1
            ).unsqueeze(-1)

            goal_dist_tensor = goal_dist_tensor[self.cont_agent_mask].reshape(
                self.num_sims, self.max_cont_agents, -1
            )
            if self.config.normalize_obs:
                goal_dist_tensor /= self.config.max_dist_to_goal
        else:
            goal_dist_tensor = torch.Tensor().to(self.device)

        # Combine the observations
        obs_all = torch.cat(
            (
                ego_state,
                partner_obs_tensor,
                map_obs_tensor,
                goal_dist_tensor,
            ),
            dim=-1,
        )

        # Only select the observations for the controlled agents
        obs_filtered = obs_all
        # obs_filtered = obs_all[self.cont_agent_mask].reshape(
        #     self.num_sims, self.max_cont_agents, -1
        # )

        return obs_filtered

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
        goal_pos = agent_info[:, 8:]
        agent_pos = agent_info[:, :2]  # x, y

        # Get minimum and maximum values for scaling
        x_min, y_min, x_max, y_max = self.get_coord_min_max(agent_info)

        # print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")

        num_agents_in_scene = np.count_nonzero(goal_pos[:, 0])

        # Draw the agent positions
        for agent_idx in range(num_agents_in_scene):

            # Use the updated scale_coord function to get centered and scaled coordinates
            current_pos_scaled = self.scale_coords(
                agent_pos[agent_idx],  # x_min, y_min, x_max, y_max
            )

            current_goal_scaled = self.scale_coords(
                goal_pos[agent_idx],  # x_min, y_min, x_max, y_max
            )

            mod_idx = agent_idx % len(COLOR_LIST)
            # if agent_idx == 0:
            pygame.draw.rect(
                surface=self.surf,
                color=COLOR_LIST[mod_idx],
                rect=pygame.Rect(
                    int(current_pos_scaled[0]),
                    int(current_pos_scaled[1]),
                    VEH_WIDTH,
                    VEH_HEIGHT,
                ),
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

            # Log
            agent_log_dict = {
                "episode_step": 90
                - self.steps_remaining[self.world_render_idx, 0, 0].item(),
                f"goal_pos/agent_{agent_idx}_goal_x": goal_pos[agent_idx, 0],
                f"goal_pos/agent_{agent_idx}_goal_y": goal_pos[agent_idx, 1],
                f"pos/agent_{agent_idx}_x": agent_pos[agent_idx, 0],
                f"pos/agent_{agent_idx}_y": agent_pos[agent_idx, 1],
                f"dist/agent_{agent_idx}_goal_dist": np.linalg.norm(
                    agent_pos[agent_idx] - goal_pos[agent_idx]
                ),
                f"pos_scaled/agent_{agent_idx}_x": current_pos_scaled[0],
                f"pos_scaled/agent_{agent_idx}_y": current_pos_scaled[1],
                f"dist_scaled/agent_{agent_idx}_goal_dist": np.linalg.norm(
                    np.array(current_pos_scaled)
                    - np.array(current_goal_scaled)
                ),
            }

            wandb.log(agent_log_dict)

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

    def scale_coords(
        self, coords, x_min=-6000, y_min=-12_000, x_max=4500, y_max=9200
    ):
        """Scale the coordinates to fit within the pygame surface window and center them.
        Args:
            coords: x, y coordinates
        """
        x, y = coords

        # Transformationn 1
        x_scaled = (x / (x_max - x_min)) * WINDOW_W
        y_scaled = (y / (y_max - y_min)) * WINDOW_H

        # Transformation 2
        # x_scaled = (x - (x_min + x_max) / 2) #* WINDOW_W
        # y_scaled = (y - (y_min + y_max) / 2) #* WINDOW_H

        return (x_scaled, y_scaled)

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        state[:, :, 0] /= self.config.max_speed
        state[:, :, 1] /= self.config.max_veh_len
        state[:, :, 2] /= self.config.max_veh_width
        state[:, :, 3] /= self.config.max_rel_goal_coords
        state[:, :, 4] /= self.config.max_rel_goal_coords

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

    def get_coord_min_max(self, coords):
        """Get the minimum and maximum values of the coordinates."""
        x_min = np.min(coords[:, 0])
        y_min = np.min(coords[:, 1])
        x_max = np.max(coords[:, 0])
        y_max = np.max(coords[:, 1])
        return x_min, y_min, x_max, y_max

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
    run = wandb.init(
        project="gpudrive",
        group="test_rendering",
    )

    wandb.define_metric("episode_step")
    # set all other agent pos logs to use this step
    wandb.define_metric("agent_*", step_metric="episode_step")

    NUM_CONT_AGENTS = 1

    env = Env(
        config=config,
        num_worlds=1,
        max_cont_agents=NUM_CONT_AGENTS,  # Number of agents to control
        data_dir="waymo_data",
        device="cuda",
    )

    obs = env.reset()
    frames = []

    for _ in range(100):

        print(f"Step: {90 - env.steps_remaining[0, 0, 0].item()}")

        # Take a random action (we're only going straight)
        rand_action = torch.Tensor(
            [[env.action_space.sample() for _ in range(NUM_CONT_AGENTS)]]
        )

        # print(
        #     f"action (acc, steer, heading): {env.action_key_to_values[rand_action.item()]}"
        # )

        # Step the environment
        obs, reward, done, info = env.step(rand_action)

        print(
            f"speed: {obs[0, 0, 0].item():.2f} | x_pos: {obs[0, 0, 3].item():.2f} | y_pos: {obs[0, 0, 4].item():.2f} | reward:{reward[0].item():.2f} | done: {done[0].item()}\n"
        )

        if done.sum() == NUM_CONT_AGENTS:
            obs = env.reset()
            print(f"RESETTING ENVIRONMENT\n")

        # frame = env.render()
        # frames.append(frame.T)

    # Log video
    # wandb.log({"scene": wandb.Video(np.array(frames), fps=10, format="gif")})
    # wandb.log({"scene": wandb.Video(np.array(frames), fps=10, format="gif")})

    run.finish()
    env.close()
