import os
import numpy as np
import gymnasium as gym
import logging
import glob
from gymnasium.spaces import Box, Discrete

from pygpudrive.env.config import RenderConfig, RenderMode
from pygpudrive.env.viz import PyGameVisualizer

import gpudrive


class AbstractMultiAgentEnv(gym.Env):
    """GPUDrive base class for multi-agent scenarios.

    Args:
        gym.Env: Gym environment class
    """

    def __init__(
        self,
        config,
    ):
        self.config = config

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

    def _set_road_reduction_params(self, params):
        """Set the road point reduction algorithm to select the k nearest
        road points within a radius. K is set in consts.hpp `kMaxAgentMapObservationsCount`.
        """
        params.observationRadius = self.config.obs_radius
        if self.config.road_obs_algorithm == "k_nearest_roadpoints":
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.KNearestEntitiesWithRadiusFiltering
            )
        else:  # Defaulting to linear algorithm
            params.roadObservationAlgorithm = (
                gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering
            )
        return params

    def _validate_data_dir(self):
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            assert False, "The data directory does not exist or is empty."

    def _setup_environment_parameters(self):
        params = gpudrive.Parameters()
        params.polylineReductionThreshold = (
            self.config.polyline_reduction_threshold
        )
        params.rewardParams = self._set_reward_params()
        params.IgnoreNonVehicles = self.config.remove_non_vehicles
        params.maxNumControlledVehicles = self.max_cont_agents
        params = self._set_collision_behavior(params)
        params = self._init_dataset(params, self.data_dir)
        params = self._set_road_reduction_params(params)
        return params

    def _initialize_simulator(self, params):
        exec_mode = (
            gpudrive.madrona.ExecMode.CPU
            if self.device == "cpu"
            else gpudrive.madrona.ExecMode.CUDA
        )
        return gpudrive.SimManager(
            exec_mode=exec_mode,
            gpu_id=0,
            num_worlds=self.num_worlds,
            json_path=self.data_dir,
            params=params,
            enable_batch_renderer=self.render_config
            and self.render_config.render_mode
            in {RenderMode.MADRONA_RGB, RenderMode.MADRONA_DEPTH},
            batch_render_view_width=self.render_config.resolution[0]
            if self.render_config
            else None,
            batch_render_view_height=self.render_config.resolution[1]
            if self.render_config
            else None,
        )

    def _setup_rendering(self):
        return PyGameVisualizer(
            self.sim, self.render_config, self.config.dist_to_goal_threshold
        )

    def _setup_controlled_agents(self):
        self.cont_agent_mask = (
            self.sim.controlled_state_tensor().to_jax() == 1
        ).squeeze(axis=2)
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )
        self.config.total_controlled_vehicles = (
            self.num_valid_controlled_agents_across_worlds
        )

    def _setup_action_space(self, action_type):
        if action_type == "discrete":
            self.action_space = self._set_discrete_action_space()
        else:
            raise ValueError(f"Action space not supported: {action_type}")

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
