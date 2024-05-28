"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import product

import wandb
import glob
import gymnasium as gym
import os

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.viz import PyGameVisualizer

# Import the simulator
import gpudrive
import logging

logging.getLogger(__name__)

#os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "./gpudrive_cache"


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
        render_mode="rgb_array",
        verbose=True,
    ):
        self.config = config

        reward_params = self._set_reward_params()

        # Configure the environment
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = 0.5
        params.rewardParams = reward_params
        params.IgnoreNonVehicles = self.config.remove_non_vehicles

        # Collision behavior
        params = self._set_collision_behavior(params)

        # Dataset initialization
        params = self._init_dataset(params)

        # Set maximum number of controlled vehicles per environment
        params.maxNumControlledVehicles = max_cont_agents

        # Choose road point reduction algorithm
        # Only returns the k nearest road points within the radius
        # K is set in consts.hpp `kMaxAgentMapObservationsCount`
        params.observationRadius = self.config.obs_radius
        if self.config.road_obs_algorithm == "k_nearest_roadpoints":
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.KNearestEntitiesWithRadiusFiltering
            )
        else:
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering
            )

        self.data_dir = data_dir
        self.num_sims = num_worlds
        self.device = device
        self.action_types = 3  # Acceleration, steering, and heading
        self.info_dim = 5

        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            assert False, "The data directory does not exist or is empty."

        # Initialize the simulator
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CPU
            if device == "cpu"
            else gpudrive.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=self.num_sims,
            json_path=self.data_dir,
            params=params,
        )

        # Rendering
        self.render_mode = render_mode
        # By default, we render the first world
        self.world_render_idx = 0
        self.visualizer = PyGameVisualizer(
            self.sim,
            self.world_render_idx,
            self.render_mode,
            self.config.dist_to_goal_threshold,
        )
        # We only want to obtain information from vehicles we control
        # By default, the sim returns information for all vehicles in a scene
        # We construct a mask to filter out the information from the non-controlled vehicles (0)
        # Mask size: (num_worlds, kMaxAgentCount)
        self.cont_agent_mask = (
            self.sim.controlled_state_tensor().to_torch() == 1
        ).squeeze(dim=2)
        # Get the indices of the controlled agents
        self.w_indices, self.k_indices = torch.where(self.cont_agent_mask)
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.max_cont_agents = max_cont_agents

        # Number of valid controlled agents across worlds (without padding agents)
        self.num_valid_controlled_agents_across_worlds = self.cont_agent_mask.sum().item()

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

        if actions is not None:
            self._apply_actions(actions)

        self.sim.step()

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

        if self.config.eval_expert_mode:
            # This is true when we are evaluating the expert performance
            info = self.sim.info_tensor().to_torch().squeeze(dim=2)

        else:  # Standard behavior: controlling vehicles
            info = (
                torch.empty(self.num_sims, self.max_agent_count, 5)
                .fill_(float("nan"))
                .to(self.device)
            )
            info[self.cont_agent_mask] = (
                self.sim.info_tensor()
                .to_torch()
                .squeeze(dim=2)
                .to(info.dtype)[self.cont_agent_mask]
            ).to(self.device)

        # if info[self.cont_agent_mask].sum().item() > 3:
        #     print("bug")

        return obs, reward, done, info

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        assert actions.shape == (
            self.num_sims,
            self.max_agent_count,
        ), """Action tensor must match the shape (num_worlds, max_agent_count)"""

        # nan actions will be ignored, but we need to replace them with zeros
        actions = torch.nan_to_num(actions, nan=0).long().to(self.device)

        # Map action indices to action values
        action_value_tensor = self.action_keys_tensor[actions]

        # Feed the actual action values to gpudrive
        self.sim.action_tensor().to_torch().copy_(action_value_tensor)

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

        self.action_keys_tensor = torch.tensor(
            [
                self.action_key_to_values[key]
                for key in sorted(self.action_key_to_values.keys())
            ]
        ).to(self.device)

        return Discrete(n=int(len(self.action_key_to_values)))

    def _set_observation_space(self) -> None:
        """Configure the observation space."""
        return Box(low=-np.inf, high=np.inf, shape=(self.get_obs().shape[-1],))

    def _set_reward_params(self):
        """Configure the reward parameters."""

        reward_params = gpudrive.RewardParams()

        if self.config.reward_type == "sparse_on_goal_achieved":
            reward_params.rewardType = gpudrive.RewardType.OnGoalAchieved
        else:
            raise ValueError(f"Invalid reward type: {self.config.reward_type}")

        # Set goal is achieved condition
        reward_params.distanceToGoalThreshold = (
            self.config.dist_to_goal_threshold
        )

        return reward_params

    def _set_collision_behavior(self, params):
        """Define what will happen when a collision occurs."""

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
        return params

    def _init_dataset(self, params):
        """Define how we sample new scenarios."""
        if self.config.sample_method == "first_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.FirstN
        elif self.config.sample_method == "random":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.Random
        elif self.config.sample_method == "pad_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.PadN
        elif self.config.sample_method == "exact_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.ExactN

        return params

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

            if self.config.norm_obs:
                ego_state_padding = self.normalize_ego_state(ego_state_padding)

        else:
            ego_state_padding = torch.Tensor().to(self.device)

        # Get patner observation
        # Partner obs: (num_worlds, kMaxAgentCount, kMaxAgentCount - 1 * num_features)
        if self.config.partner_obs:
            full_partner_obs = (
                self.sim.partner_observations_tensor().to_torch()
            )
            if self.config.norm_obs:  # Normalize observations and then flatten
                full_partner_obs = self.normalize_and_flatten_partner_obs(
                    full_partner_obs
                )
            else:  # Flatten along the last two dimensions
                full_partner_obs = full_partner_obs.flatten(start_dim=2)

            # Pad with nans
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

        else:
            partner_obs_padding = torch.Tensor().to(self.device)

        # Get road map
        # Roadmap obs: (num_worlds, kMaxAgentCount, kMaxRoadEntityCount, num_features)
        # Flatten over the last two dimensions to get (num_worlds, kMaxAgentCount, kMaxRoadEntityCount * num_features)
        if self.config.road_map_obs:

            full_map_obs = self.sim.agent_roadmap_tensor().to_torch()

            if self.config.norm_obs:
                full_map_obs = self.normalize_and_flatten_map_obs(full_map_obs)
            else:
                full_map_obs = full_map_obs.flatten(start_dim=2)

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
        return self.visualizer.draw(self.cont_agent_mask)

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= self.config.max_speed
        state[:, :, 1] /= self.config.max_veh_len
        state[:, :, 2] /= self.config.max_veh_width

        # Relative goal coordinates
        state[:, :, 3] = self._norm(
            state[:, :, 3],
            self.config.min_rel_goal_coord,
            self.config.max_rel_goal_coord,
        )
        state[:, :, 4] = self._norm(
            state[:, :, 4],
            self.config.min_rel_goal_coord,
            self.config.max_rel_goal_coord,
        )

        return state

    def normalize_and_flatten_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: torch.Tensor of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        """

        # TODO: Fix (there should not be nans in the obs)
        # BUG: remove nan values
        obs = torch.nan_to_num(obs, nan=0)

        # Speed
        obs[:, :, :, 0] /= self.config.max_speed

        # Relative position
        obs[:, :, :, 1] = self._norm(
            obs[:, :, :, 1],
            self.config.min_rel_agent_pos,
            self.config.max_rel_agent_pos,
        )
        obs[:, :, :, 2] = self._norm(
            obs[:, :, :, 2],
            self.config.min_rel_agent_pos,
            self.config.max_rel_agent_pos,
        )

        # Orientation
        obs[:, :, :, 3] /= self.config.max_orientation_rad

        # Vehicle length and width
        obs[:, :, :, 4] /= self.config.max_veh_len
        obs[:, :, :, 5] /= self.config.max_veh_width

        # Object type
        # TODO: One hot encode

        return obs.flatten(start_dim=2)

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""

        # Position coordinates
        obs[:, :, :, 0] = self._norm(
            obs[:, :, :, 0],
            self.config.min_rm_coord,
            self.config.max_rm_coord,
        )

        obs[:, :, :, 1] = self._norm(
            obs[:, :, :, 1],
            self.config.min_rm_coord,
            self.config.max_rm_coord,
        )

        # Orientation
        obs[:, :, :, 2] /= self.config.max_orientation_rad

        # TODO: Type of road entity
        # Remove for now
        obs = obs[:, :, :, :3]

        return obs.flatten(start_dim=2)

    def _norm(self, x, min_val, max_val):
        """Normalize a value between -1 and 1."""
        return 2 * ((x - min_val) / (max_val - min_val)) - 1

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
                f"Total number of controlled agents across scenes: {self.num_valid_controlled_agents_across_worlds}"
            )
            logging.info("----------------------\n")

    @property
    def steps_remaining(self):
        return self.sim.steps_remaining_tensor().to_torch()[0][0].item()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    config = EnvConfig(
        ego_state=True,
        partner_obs=True,
        road_map_obs=True,
        norm_obs=True,
    )

    NUM_CONT_AGENTS = 128
    NUM_WORLDS = 1

    env = Env(
        config=config,
        num_worlds=NUM_WORLDS,
        max_cont_agents=NUM_CONT_AGENTS,
        num_worlds=1,
        max_cont_agents=NUM_CONT_AGENTS,  # Number of agents to control
        data_dir="waymo_data_repeat",
        device="cuda",
        render_mode="rgb_array",
    )

    obs = env.reset()
    frames = []

    for _ in range(200):

        print(f"Step: {91 - env.steps_remaining}")

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

        if env.steps_remaining == 0:
            obs = env.reset()
