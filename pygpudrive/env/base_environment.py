"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import product

# import wandb
import glob
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import random
import os
import math

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.viz import PyGameVisualizer

# Import the simulator
import gpudrive
import logging

logging.getLogger(__name__)


class Env(gym.Env):
    """
    GPU Drive Gym Environment.
    """

    metadata = {
        "render_mode": [
            "pygame_absolute",
            "pygame_egocentric",
            "madrona_rgb",
            "madrona_depth"
        ], 
        "pygame_option":[
            "human",
            "rgb"
        ],
        "madrona_option":[
            "agent_view",
            "top_down"
        ],
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
        render_options: dict = None, # sample {"render_mode": "madrona_rgb", "view_mode": "agent_view", "resolution": (128, 128)}
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
        params.polylineReductionThreshold = 1.0
        params.observationRadius = 10.0
        params.rewardParams = reward_params
        if render_options is not None:
            params.enable_batch_renderer = render_options['render_mode'].startswith("madrona")
            params.batch_render_view_width = render_options['resolution'][0]
            params.batch_render_view_height = render_options['resolution'][1]

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

        # Set maximum number of controlled vehicles per environment
        params.maxNumControlledVehicles = max_cont_agents

        self.data_dir = data_dir
        self.num_sims = num_worlds
        self.device = device
        self.action_types = 3  # Acceleration, steering, and heading

        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            assert False, "The data directory does not exist or is empty."

        # Initialize the simulator
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CPU
            if device == "cpu"
            else gpudrive.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=self.num_sims,
            auto_reset=True,
            json_path=self.data_dir,
            params=params,
        )

        # Rendering
        self.render_options = render_options
        self.world_render_idx = 0
        agent_count = (
            self.sim.shape_tensor().to_torch()[self.world_render_idx, :][0].item()
        )
        self.visualizer = PyGameVisualizer(self.sim, self.world_render_idx, self.render_options, self.config.dist_to_goal_threshold)

        # We only want to obtain information from vehicles we control
        # By default, the sim returns information for all vehicles in a scene
        # We construct a mask to filter out the information from the expert controlled vehicles (0)
        # Mask size: (num_worlds, kMaxAgentCount)
        self.cont_agent_mask = (
            self.sim.controlled_state_tensor().to_torch() == 1
        ).squeeze(dim=2)
        # Get the indices of the controlled agents
        self.w_indices, self.k_indices = torch.where(self.cont_agent_mask)
        self.max_agent_count = self.cont_agent_mask.shape[1]
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
        """Take simultaneous actions for each controlled agent in all `num_worlds` environments.

        Args:
            actions (torch.Tensor): The action indices for all agents in all worlds.
        """
        assert actions.shape == (
            self.num_sims,
            self.max_agent_count,
        ), """Action tensor must match the shape (num_worlds, max_agent_count)"""

        # Convert nan values to 0 for dead agents.
        # These actions will be ignored, their value does not matter (except it cannot be nan)
        actions = torch.nan_to_num(actions, nan=0)

        # Convert action indices to action values
        actions_shape = self.sim.action_tensor().to_torch().shape[1]
        action_value_tensor = torch.zeros(
            (self.num_sims, actions_shape, self.action_types)
        )

        # GPU Drive expects a tensor of shape (num_worlds, kMaxAgentCount, 3)
        # We insert the actions for the controlled vehicles, the others will be ignored
        for world_idx in range(self.num_sims):
            for agent_idx in range(self.max_agent_count):
                action_idx = actions[world_idx, agent_idx].item()
                action_value_tensor[world_idx, agent_idx, :] = torch.Tensor(
                    self.action_key_to_values[action_idx]
                )

        # Feed the actual action values to gpudrive
        self.sim.action_tensor().to_torch().copy_(action_value_tensor)

        # Step the simulator
        self.sim.step()

        # Obtain the next observations, rewards, and done flags
        obs = self.get_obs()

        reward = (
            torch.empty(self.num_sims, self.max_agent_count)
            .fill_(float("nan"))
            .to(self.device)
        )
        reward[self.cont_agent_mask] = (
            self.sim.reward_tensor()
            .to_torch()
            .squeeze(dim=2)[self.cont_agent_mask]
        )

        done = (
            torch.empty(self.num_sims, self.max_agent_count)
            .fill_(float("nan"))
            .to(self.device)
        )
        done[self.cont_agent_mask] = (
            self.sim.done_tensor()
            .to_torch()
            .squeeze(dim=2)
            .to(done.dtype)[self.cont_agent_mask]
        )

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
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        # Get the ego state
        # Ego state: (num_worlds, kMaxAgentCount, features)
        if self.config.ego_state:
            ego_state_padding = (
                torch.empty(self.num_sims, self.max_agent_count, 6)
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
            full_partner_obs = (
                self.sim.partner_observations_tensor()
                .to_torch()
                .flatten(start_dim=2)
            )

            partner_obs_padding = (
                torch.empty(
                    self.num_sims,
                    self.max_agent_count,
                    full_partner_obs.shape[2],
                )
                .fill_(float("nan"))
                .to(self.device)
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
            full_map_obs = (
                self.sim.agent_roadmap_tensor().to_torch().flatten(start_dim=2)
            )

            map_obs_padding = (
                torch.empty(
                    self.num_sims, self.max_agent_count, full_map_obs.shape[2]
                )
                .fill_(float("nan"))
                .to(self.device)
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
        if (self.render_mode == "madrona"):
            return self.sim.rgb_tensor().to_torch()

        return self.visualizer.draw(self.cont_agent_mask)
        

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
                f"Total number of controlled agents across scenes: {self.num_valid_controlled_agents}"
            )
            logging.info("----------------------\n")

    @property
    def steps_remaining(self):
        return self.sim.steps_remaining_tensor().to_torch()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    config = EnvConfig(
        partner_obs=True,
        road_map_obs=True,
    )
    # run = wandb.init(
    #     project="gpudrive",
    #     group="test_rendering",
    # )
    NUM_CONT_AGENTS = 0
    NUM_WORLDS = 3

    env = Env(
        config=config,
        num_worlds=1,
        auto_reset=False,
        max_cont_agents=NUM_CONT_AGENTS,  # Number of agents to control
        data_dir="/home/aarav/gpudrive/nocturne_data",
        device="cuda",
        render_mode="rgb_array",
    )
    
    obs = env.reset()
    frames = []

    for _ in range(100):
        print(f"Step: {90 - env.steps_remaining[0, 2, 0].item()}")

        # Take a random action (we're only going straight)
        # rand_action = torch.Tensor(
        #     [
        #         [
        #             env.action_space.sample()
        #             for _ in range(NUM_CONT_AGENTS * NUM_WORLDS)
        #         ]
        #     ]
        # ).reshape(NUM_WORLDS, NUM_CONT_AGENTS)

        # # Step the environment
        # obs, reward, done, info = env.step(rand_action)

        # if done.sum() == NUM_CONT_AGENTS:
        #     obs = env.reset()
        #     print(f"RESETTING ENVIRONMENT\n")
        env.sim.step()
        frame = env.render()
        frames.append(frame)

    import imageio
    imageio.mimsave("out.gif", frames)
    # Log video
    # wandb.log({"scene": wandb.Video(np.array(frames), fps=10, format="gif")})
    # wandb.log({"scene": wandb.Video(np.array(frames), fps=10, format="gif")})

    # run.finish()
    env.visualizer.destroy()
