"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
from itertools import product

import jax
import jax.numpy as jnp

import glob
import gymnasium as gym
import os

from pygpudrive.env.config import *
from pygpudrive.env.viz import PyGameVisualizer

# Import the simulator
import gpudrive
import logging

logging.getLogger(__name__)


class Env(gym.Env):
    """
    GPU Drive Gym Environment.
    """

    def __init__(
        self,
        config,
        num_worlds,
        max_cont_agents,
        data_dir,
        device="cuda",
        render_config: RenderConfig = RenderConfig(),
        action_type="discrete",
        verbose=True,
    ):
        self.config = config

        reward_params = self._set_reward_params()

        # Configure the environment
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = 1.0
        params.rewardParams = reward_params
        params.IgnoreNonVehicles = self.config.remove_non_vehicles

        # Collision behavior
        params = self._set_collision_behavior(params)

        # Dataset initialization
        params = self._init_dataset(params, data_dir)

        # Set maximum number of controlled vehicles per environment
        params.maxNumControlledVehicles = max_cont_agents

        # Set road point observation radius and reduction algorithm
        params = self._set_road_reduction_params(params)

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
            enable_batch_renderer=render_config is not None
            and render_config.render_mode
            in {RenderMode.MADRONA_RGB, RenderMode.MADRONA_DEPTH},
            batch_render_view_width=render_config.resolution[0]
            if render_config is not None
            else None,
            batch_render_view_height=render_config.resolution[1]
            if render_config is not None
            else None,
        )

        # Rendering
        self.render_config = render_config
        self.world_render_idx = 0
        self.visualizer = PyGameVisualizer(
            self.sim, self.render_config, self.config.dist_to_goal_threshold
        )

        # We only want to obtain information from vehicles we control
        # By default, the sim returns information for all vehicles in a scene
        # We construct a mask to filter out the information from the non-controlled vehicles (0)
        # Mask size: (num_worlds, kMaxAgentCount)
        self.cont_agent_mask = (
            self.sim.controlled_state_tensor().to_jax() == 1
        ).squeeze(axis=2)

        # Get the indices of the controlled agents
        self.w_indices, self.k_indices = jnp.where(self.cont_agent_mask)
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.max_cont_agents = max_cont_agents

        # Number of valid controlled agents across worlds (without padding agents)
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Update the config with the relevant number of controlled agents
        self.config.total_controlled_vehicles = (
            self.num_valid_controlled_agents_across_worlds
        )

        # Set up action space
        if action_type == "discrete":
            self.action_space = self._set_discrete_action_space()
        else:
            raise ValueError(f"Action space not supported: {action_type}")

        # Set observation space
        self.observation_space = self._set_observation_space()
        self.obs_dim = self.observation_space.shape[0]

        self._print_info(verbose)

    def reset(self):
        """Reset the worlds and return the initial observations."""
        for sim_idx in range(self.num_sims):
            self.sim.reset(sim_idx)

        return self.get_obs()

    def get_dones(self):
        """Get dones for all agents."""
        return self.sim.done_tensor().to_jax().squeeze(axis=2)

    def get_infos(self):
        """Get info for all agents."""
        if self.config.eval_expert_mode:
            # This is true when we are evaluating the expert performance
            info = self.sim.info_tensor().to_jax().squeeze(dim=2)

        else:  # Standard behavior: controlling vehicles
            info = (self.sim.info_tensor().to_jax().squeeze(dim=2)).to(
                self.device
            )
        return info

    def get_rewards(self):
        """Get rewards for all agents."""
        return self.sim.reward_tensor().to_jax().squeeze(axis=2)

    def step_dynamics(self, actions):
        """Step the simulator."""
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        assert actions.shape == (
            self.num_sims,
            self.max_agent_count,
        ), """Action tensor must match the shape (num_worlds, max_agent_count)"""

        # Nan actions will be ignored, but we need to replace them with zeros
        actions = jnp.nan_to_num(actions, nan=0)
        # TODO: Ensure actions are on GPU -- jax.device_put(actions, device=self.device)

        # Map action indices to action values
        action_value_tensor = self.action_keys_tensor[actions]

        # Feed the actual action values to gpudrive
        self.sim.action_tensor().to_jax().at[:, :, :].set(action_value_tensor)

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space."""

        self.steer_actions = jnp.asarray(self.config.steer_actions)
        self.accel_actions = jnp.asarray(self.config.accel_actions)
        self.head_actions = jnp.array([0])

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

        self.action_keys_tensor = jnp.array(
            [
                self.action_key_to_values[key]
                for key in sorted(self.action_key_to_values.keys())
            ]
        )

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

    def _init_dataset(self, params, data_dir):
        """Define how we sample new scenarios."""

        if self.config.sample_method == "first_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.FirstN
        elif self.config.sample_method == "random_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.RandomN
        elif self.config.sample_method == "pad_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.PadN
        elif self.config.sample_method == "exact_n":
            params.datasetInitOptions = gpudrive.DatasetInitOptions.ExactN

        self.data_dir = data_dir

        return params

    def get_obs(self):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            jnp.array: (num_worlds, max_agent_count, num_features)
        """

        # Get the ego state
        # Ego state: (num_worlds, kMaxAgentCount, features)
        if self.config.ego_state:
            ego_states = self.sim.self_observation_tensor().to_jax()
            if self.config.norm_obs:
                ego_states = self.normalize_ego_state(ego_states)
        else:
            ego_states = jnp.array()

        # Get patner observation
        # Partner obs: (num_worlds, kMaxAgentCount, kMaxAgentCount - 1 * num_features)
        if self.config.partner_obs:
            partner_observations = (
                self.sim.partner_observations_tensor().to_jax()
            )
            if self.config.norm_obs:  # Normalize observations and then flatten
                partner_observations = self.normalize_and_flatten_partner_obs(
                    partner_observations
                )
            else:  # Flatten along the last two dimensions
                partner_observations = partner_observations.flatten(
                    start_dim=2
                )

        else:
            partner_observations = jnp.array()

        # Get road map
        # Roadmap obs: (num_worlds, kMaxAgentCount, kMaxRoadEntityCount, num_features)
        # Flatten over the last two dimensions to get (num_worlds, kMaxAgentCount, kMaxRoadEntityCount * num_features)
        if self.config.road_map_obs:

            road_map_observations = self.sim.agent_roadmap_tensor().to_jax()

            if self.config.norm_obs:
                road_map_observations = self.normalize_and_flatten_map_obs(
                    road_map_observations
                )
            else:
                road_map_observations = road_map_observations.flatten(
                    start_dim=2
                )
        else:
            road_map_observations = jnp.array()

        # Combine the observations
        obs_filtered = jnp.concatenate(
            (
                ego_states,
                partner_observations,
                road_map_observations,
            ),
            axis=-1,
        )

        return obs_filtered

    def render(self, world_render_idx=0):
        if world_render_idx >= self.num_sims:
            # Raise error but dont interrupt the training
            print(f"Invalid world_render_idx: {world_render_idx}")
            return None
        if self.render_config.render_mode in {
            RenderMode.PYGAME_ABSOLUTE,
            RenderMode.PYGAME_EGOCENTRIC,
            RenderMode.PYGAME_LIDAR,
        }:
            return self.visualizer.getRender(
                world_render_idx=world_render_idx,
                cont_agent_mask=self.cont_agent_mask,
            )
        elif self.render_config.render_mode in {
            RenderMode.MADRONA_RGB,
            RenderMode.MADRONA_DEPTH,
        }:
            return self.visualizer.getRender()

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state.at[:, :, 0].divide(self.config.max_speed)
        state.at[:, :, 1].divide(self.config.max_veh_len)
        state.at[:, :, 2].divide(self.config.max_veh_width)

        # Relative goal coordinates
        state.at[:, :, 3].set(
            self._norm(
                state[:, :, 3],
                self.config.min_rel_goal_coord,
                self.config.max_rel_goal_coord,
            )
        )

        state.at[:, :, 4].set(
            self._norm(
                state[:, :, 4],
                self.config.min_rel_goal_coord,
                self.config.max_rel_goal_coord,
            )
        )

        # Uncommment this to exclude the collision state
        # (1 if vehicle is in collision, 1 otherwise)
        # state = state[:, :, :5]

        return state

    def normalize_and_flatten_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: jnp.array of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        """

        # Speed
        obs.at[:, :, :, 0].divide(self.config.max_speed)

        # Relative position
        obs.at[:, :, :, 1].set(
            self._norm(
                obs[:, :, :, 1],
                self.config.min_rel_agent_pos,
                self.config.max_rel_agent_pos,
            )
        )
        obs.at[:, :, :, 2].set(
            self._norm(
                obs[:, :, :, 2],
                self.config.min_rel_agent_pos,
                self.config.max_rel_agent_pos,
            )
        )

        # Orientation (heading)
        obs.at[:, :, :, 3].divide(self.config.max_orientation_rad)

        # Vehicle length and width
        obs.at[:, :, :, 4].divide(self.config.max_veh_len)
        obs.at[:, :, :, 5].divide(self.config.max_veh_width)

        # Hot-encode object type
        shifted_type_obs = obs[:, :, :, 6] - 6
        type_indices = jnp.where(
            shifted_type_obs >= 0,
            shifted_type_obs,
            0,
        )
        one_hot_object_type = jax.nn.one_hot(
            type_indices,
            num_classes=4,
        )

        # Concatenate the one-hot encoding with the rest of the features
        obs = jnp.concat((obs[:, :, :, :6], one_hot_object_type), axis=-1)

        return obs.reshape(self.num_sims, self.max_agent_count, -1)

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""

        # Road point coordinates
        obs.at[:, :, :, 0].set(
            self._norm(
                obs[:, :, :, 0],
                self.config.min_rm_coord,
                self.config.max_rm_coord,
            )
        )

        obs.at[:, :, :, 1].set(
            self._norm(
                obs[:, :, :, 1],
                self.config.min_rm_coord,
                self.config.max_rm_coord,
            )
        )

        # Road line segment length
        obs.at[:, :, :, 2].divide(self.config.max_road_line_segmment_len)

        # Road scale (width and height)
        obs.at[:, :, :, 3].divide(self.config.max_road_scale)
        # obs.at[:, :, :, 4] seems already scaled

        # Road point orientation
        obs.at[:, :, :, 5].divide(self.config.max_orientation_rad)

        # Road types: one-hot encode them
        one_hot_road_type = jax.nn.one_hot(obs[:, :, :, 6], num_classes=7)

        # Concatenate the one-hot encoding with the rest of the features (exclude index 3 and 4)
        obs = jnp.concatenate((obs[:, :, :, :6], one_hot_road_type), axis=-1)

        return obs.reshape(self.num_sims, self.max_agent_count, -1)

    def _norm(self, x, min_val, max_val):
        """Normalize a value between -1 and 1."""
        return 2 * ((x - min_val) / (max_val - min_val)) - 1

    def _set_road_reduction_params(self, params):
        """Set the road point reduction algorithm to select the k nearest
        road points within a radius. K is set in consts.hpp `kMaxAgentMapObservationsCount`.
        """
        params.observationRadius = self.config.obs_radius
        if self.config.road_obs_algorithm == "k_nearest_roadpoints":
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.KNearestEntitiesWithRadiusFiltering
            )
        else:
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering
            )
        return params

    def _print_info(self, verbose=True):
        """Print initialization information."""
        if verbose:
            logging.info("----------------------")
            logging.info(f"Device: {self.device}")
            logging.info(f"Number of worlds: {self.num_sims}")
            logging.info(
                f"Number of maps in data directory: {len(glob.glob(f'{self.data_dir}/*.json')) - 1}"
            )
            logging.info(
                f"Total number of controlled agents across scenes: {self.num_valid_controlled_agents_across_worlds}"
            )
            logging.info(f"using {self.config.road_obs_algorithm}")
            logging.info("----------------------\n")

    @property
    def steps_remaining(self):
        return self.sim.steps_remaining_tensor().to_jax()[0][0].item()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    env_config = EnvConfig(sample_method="first_n")
    render_config = RenderConfig()

    TOTAL_STEPS = 1000
    MAX_NUM_OBJECTS = 128
    NUM_WORLDS = 10

    env = Env(
        config=env_config,
        num_worlds=NUM_WORLDS,
        max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
        data_dir="formatted_json_v2_no_tl_train",
        device="cuda",
        render_config=render_config,
    )

    obs = env.reset()

    for _ in range(TOTAL_STEPS):

        # Take a random actions
        rand_action = jax.random.randint(
            key=jax.random.PRNGKey(0),
            shape=(NUM_WORLDS, MAX_NUM_OBJECTS),
            minval=0,
            maxval=env.action_space.n,
        )

        # Step the environment
        env.step_dynamics(rand_action)

        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

        if done.any():
            print("Done")
            obs = env.reset()

    # run.finish()
    env.visualizer.destroy()
