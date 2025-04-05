"""Torch Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
from itertools import product
import mediapy as media
import gymnasium

from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
    LidarObs,
    BevObs,
)

from gpudrive.env import constants
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.base_env import GPUDriveGymEnv
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.roadgraph import (
    LocalRoadGraphPoints,
    GlobalRoadGraphPoints,
)
from gpudrive.datatypes.metadata import Metadata
from gpudrive.datatypes.info import Info

from gpudrive.visualize.core import MatplotlibVisualizer
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.geometry import normalize_min_max

from gpudrive.integrations.vbd.data_utils import process_scenario_data


class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        data_loader,
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
        backend="torch",
    ):
        # Initialization of environment configurations
        self.config = config
        self.data_loader = data_loader
        self.num_worlds = data_loader.batch_size
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config
        self.backend = backend
        self.max_num_agents_in_scene = self.config.max_num_agents_in_scene
        self.world_time_steps = torch.zeros(
            self.num_worlds, dtype=torch.short, device=self.device
        )

        # Initialize reward weights tensor if using reward_conditioned
        self.reward_weights_tensor = None
        if (
            hasattr(self.config, "reward_type")
            and self.config.reward_type == "reward_conditioned"
        ):
            # Use default condition_mode from config or fall back to "random"
            condition_mode = getattr(self.config, "condition_mode", "random")
            agent_type = getattr(self.config, "agent_type", None)
            self._set_reward_weights(
                condition_mode=condition_mode, agent_type=agent_type
            )

        # Environment parameter setup
        params = self._setup_environment_parameters()

        # Initialize the iterator once
        self.data_iterator = iter(self.data_loader)

        # Get the initial data batch (set of traffic scenarios)
        self.data_batch = next(self.data_iterator)

        # Initialize simulator
        self.sim = self._initialize_simulator(params, self.data_batch)

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        self.episode_len = self.config.episode_len

        # Initialize VBD model if used
        self._initialize_vbd()

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.get_obs(self.cont_agent_mask).shape[-1],),
        )

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_space.shape[-1],),
            dtype=np.float32,
        )

        self._setup_action_space(action_type)
        self.single_action_space = self.action_space

        self.num_agents = self.cont_agent_mask.sum().item()

        # Rendering setup
        self.vis = MatplotlibVisualizer(
            sim_object=self.sim,
            controlled_agent_mask=self.cont_agent_mask,
            goal_radius=self.config.dist_to_goal_threshold,
            backend=self.backend,
            num_worlds=self.num_worlds,
            render_config=self.render_config,
            env_config=self.config,
        )

    def _initialize_vbd(self):
        """
        Initialize the Versatile Behavior Diffusion (VBD) model and related
        components. Link: https://arxiv.org/abs/2404.02524.

        Args:
            config: Configuration object containing VBD settings.
        """
        self.use_vbd = self.config.use_vbd
        self.vbd_trajectory_weight = self.config.vbd_trajectory_weight

        # Set initialization steps - ensure minimum steps for VBD
        if self.use_vbd:
            self.init_steps = max(
                self.config.init_steps, 10
            )  # Minimum 10 steps for VBD
        else:
            self.init_steps = self.config.init_steps

        if (
            self.use_vbd
            and hasattr(self.config, "vbd_model_path")
            and self.config.vbd_model_path
        ):
            self.vbd_model = self._load_vbd_model(self.config.vbd_model_path)

            self.vbd_trajectories = torch.zeros(
                (
                    self.num_worlds,
                    self.max_agent_count,
                    self.episode_len - self.init_steps,
                    5,
                ),
                device=self.device,
                dtype=torch.float32,
            )

            self._generate_vbd_trajectories()
        else:
            self.vbd_model = None
            self.vbd_trajectories = None

    def _load_vbd_model(self, model_path):
        """Load the Versatile Behavior Diffusion (VBD) model from checkpoint."""
        from gpudrive.integrations.vbd.sim_agent.sim_actor import VBDTest

        model = VBDTest.load_from_checkpoint(
            model_path, torch.device(self.device)
        )
        _ = model.eval()
        return model

    def _generate_sample_batch(self, init_steps=10):
        """Generate a sample batch for the VBD model."""
        means_xy = (
            self.sim.world_means_tensor().to_torch()[:, :2].to(self.device)
        )

        # Get the logged trajectory and restore the mean
        log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
        )
        log_trajectory.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )

        # Get global road graph and restore the mean
        global_road_graph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        global_road_graph.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )
        global_road_graph.restore_xy()

        # Get global agent observations and restore the mean
        global_agent_obs = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=self.sim.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        global_agent_obs.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )
        metadata = Metadata.from_tensor(
            metadata_tensor=self.sim.metadata_tensor(),
            backend=self.backend,
        )
        sample_batch = process_scenario_data(
            max_controlled_agents=self.max_cont_agents,
            controlled_agent_mask=self.cont_agent_mask,
            global_agent_obs=global_agent_obs,
            global_road_graph=global_road_graph,
            log_trajectory=log_trajectory,
            episode_len=self.episode_len,
            init_steps=init_steps,
            raw_agent_types=self.sim.info_tensor().to_torch()[:, :, 4],
            metadata=metadata,
        )
        return sample_batch

    def _set_reward_weights(
        self, env_idx_list=None, condition_mode="random", agent_type=None
    ):
        """Set agent reward weights for all or specific environments.

        Args:
            env_idx_list: List of environment indices to generate new weights for.
                        If None, all environments are updated.
            condition_mode: Determines how reward weights are sampled:
                        - "random": Random sampling within bounds (default for training)
                        - "fixed": Use predefined agent_type weights (for testing)
                        - "preset": Use a specific preset from agent_type parameter
            agent_type: Specifies which preset weights to use if condition_mode is "preset" or "fixed"
                    If condition_mode is "preset", can be one of: "cautious", "aggressive", "balanced"
                    If condition_mode is "fixed", should be a tensor of shape [3] with weight values
        """
        if self.reward_weights_tensor is None:
            self.reward_weights_tensor = torch.zeros(
                self.num_worlds,
                self.max_cont_agents,
                3,  # collision, goal_achieved, off_road
                device=self.device,
            )

        # Read bounds for the three reward components
        lower_bounds = torch.tensor(
            [
                self.config.collision_weight_lb,
                self.config.goal_achieved_weight_lb,
                self.config.off_road_weight_lb,
            ],
            device=self.device,
        )

        upper_bounds = torch.tensor(
            [
                self.config.collision_weight_ub,
                self.config.goal_achieved_weight_ub,
                self.config.off_road_weight_ub,
            ],
            device=self.device,
        )
        bounds_range = upper_bounds - lower_bounds

        # Preset agent personality types
        agent_presets = {
            "cautious": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.9,  # Strong collision penalty
                    self.config.goal_achieved_weight_ub
                    * 0.7,  # Moderate goal reward
                    self.config.off_road_weight_lb
                    * 0.9,  # Strong off-road penalty
                ],
                device=self.device,
            ),
            "aggressive": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.5,  # Lower collision penalty
                    self.config.goal_achieved_weight_ub
                    * 0.9,  # Higher goal reward
                    self.config.off_road_weight_lb
                    * 0.6,  # Moderate off-road penalty
                ],
                device=self.device,
            ),
            "balanced": torch.tensor(
                [
                    (
                        self.config.collision_weight_lb
                        + self.config.collision_weight_ub
                    )
                    / 2,
                    (
                        self.config.goal_achieved_weight_lb
                        + self.config.goal_achieved_weight_ub
                    )
                    / 2,
                    (
                        self.config.off_road_weight_lb
                        + self.config.off_road_weight_ub
                    )
                    / 2,
                ],
                device=self.device,
            ),
            "risk_taker": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.3,  # Minimal collision penalty
                    self.config.goal_achieved_weight_ub,  # Maximum goal reward
                    self.config.off_road_weight_lb
                    * 0.4,  # Low off-road penalty
                ],
                device=self.device,
            ),
        }

        # Determine which environments to update
        if env_idx_list is None:
            env_idx_list = list(range(self.num_worlds))

        env_indices = torch.tensor(env_idx_list, device=self.device)
        num_envs = len(env_indices)

        if condition_mode == "random":
            # Traditional random sampling within bounds
            random_values = torch.rand(
                num_envs, self.max_cont_agents, 3, device=self.device
            )
            scaled_values = lower_bounds + random_values * bounds_range

        elif condition_mode == "preset":
            # Use a predefined agent type
            if agent_type not in agent_presets:
                raise ValueError(
                    f"Unknown agent_type: {agent_type}. Available types: {list(agent_presets.keys())}"
                )

            # Create a tensor with the preset weights for all agents in the specified environments
            preset_weights = agent_presets[agent_type]
            scaled_values = (
                preset_weights.unsqueeze(0)
                .unsqueeze(0)
                .expand(num_envs, self.max_cont_agents, 3)
            )

        elif condition_mode == "fixed":
            # Use custom provided weights
            if agent_type is None or not isinstance(agent_type, torch.Tensor):
                raise ValueError(
                    "For condition_mode='fixed', agent_type must be a tensor of shape [3]"
                )

            custom_weights = agent_type.to(device=self.device)
            if custom_weights.shape != (3,):
                raise ValueError(
                    f"agent_type tensor must have shape [3], got {custom_weights.shape}"
                )

            scaled_values = (
                custom_weights.unsqueeze(0)
                .unsqueeze(0)
                .expand(num_envs, self.max_cont_agents, 3)
            )

        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        # Update the weights tensor for the specified environments
        self.reward_weights_tensor[env_indices.cpu()] = scaled_values

        return self.reward_weights_tensor

    def reset(
        self,
        mask=None,
        env_idx_list=None,
        condition_mode=None,
        agent_type=None,
    ):
        """Reset the worlds and return the initial observations.

        Args:
            mask: Optional mask indicating which agents to return observations for
            env_idx_list: Optional list of environment indices to reset
            condition_mode: Determines how reward weights are sampled:
                        - "random": Random sampling within bounds (default for training)
                        - "fixed": Use predefined agent_type weights (for testing)
                        - "preset": Use a specific preset from agent_type parameter
            agent_type: Specifies which preset weights to use or custom weights
        """
        if env_idx_list is not None:
            self.sim.reset(env_idx_list)
        else:
            env_idx_list = list(range(self.num_worlds))
            self.sim.reset(env_idx_list)

        # Re-initialize reward weights if using reward_conditioned
        if (
            hasattr(self.config, "reward_type")
            and self.config.reward_type == "reward_conditioned"
        ):
            # Use the specified condition_mode or default to the config setting
            mode = (
                condition_mode
                if condition_mode is not None
                else getattr(self.config, "condition_mode", "random")
            )
            self._set_reward_weights(
                env_idx_list, condition_mode=mode, agent_type=agent_type
            )

        self.world_time_steps.zero_()

        # Advance the simulator with log playback if warmup steps are provided
        if self.init_steps > 0:
            self.advance_sim_with_log_playback(
                init_steps=self.init_steps,
                # render_init=self.render_config.render_init,
            )

        return self.get_obs(mask)

    def get_dones(self):
        return (
            self.sim.done_tensor()
            .to_torch()
            .clone()
            .squeeze(dim=2)
            .to(torch.float)
        )

    def get_infos(self):
        return Info.from_tensor(
            self.sim.info_tensor(),
            backend=self.backend,
            device=self.device,
        )

    def get_rewards(
        self,
        collision_weight=-0.5,
        goal_achieved_weight=1.0,
        off_road_weight=-0.5,
        world_time_steps=None,
        log_distance_weight=0.01,
    ):
        """Obtain the rewards for the current step.
        By default, the reward is a weighted combination of the following components:
        - collision
        - goal_achieved
        - off_road

        The importance of each component is determined by the weights.
        """

        # Return the weighted combination of the reward components
        info_tensor = self.sim.info_tensor().to_torch().clone()
        off_road = info_tensor[:, :, 0].to(torch.float)

        # True if the vehicle is in collision with another road object
        # (i.e. a cyclist or pedestrian)
        collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
        goal_achieved = info_tensor[:, :, 3].to(torch.float)

        if self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

        elif self.config.reward_type == "weighted_combination":
            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            return weighted_rewards

        elif self.config.reward_type == "reward_conditioned":
            # Extract individual weight components from the tensor
            # Shape: [num_worlds, max_agents, 3]
            if self.reward_weights_tensor is None:
                self._set_reward_weights()

            # Apply the weights in a vectorized manner
            # Each index in dimension 2 corresponds to a specific weight:
            # 0: collision, 1: goal_achieved, 2: off_road
            weighted_rewards = (
                self.reward_weights_tensor[:, :, 0] * collided
                + self.reward_weights_tensor[:, :, 1] * goal_achieved
                + self.reward_weights_tensor[:, :, 2] * off_road
            )

            return weighted_rewards

        elif self.config.reward_type == "distance_to_vdb_trajs":
            # Reward based on distance to VBD predicted trajectories
            # (i.e. the deviation from the predicted trajectory)
            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            agent_states = GlobalEgoState.from_tensor(
                self.sim.absolute_self_observation_tensor(),
                self.backend,
                self.device,
            )

            agent_pos = torch.stack(
                [agent_states.pos_x, agent_states.pos_y], dim=-1
            )

            # Extract VBD positions at current time steps for each world
            vbd_pos = []
            for i in range(self.num_worlds):
                current_time = (
                    self.world_time_steps[i].item() - self.init_steps
                )
                # Make sure we don't exceed trajectory length
                current_time = min(
                    current_time, self.vbd_trajectories.shape[2] - 1
                )
                vbd_pos.append(self.vbd_trajectories[i, :, current_time, :2])
            vbd_pos_tensor = torch.stack(vbd_pos)

            # Compute euclidean distance between agent and logs
            dist_to_vbd = torch.norm(vbd_pos_tensor - agent_pos, dim=-1)

            # Add reward based on inverse distance to logs
            weighted_rewards += self.vbd_trajectory_weight * torch.exp(
                -dist_to_vbd
            )

            return weighted_rewards

        elif self.config.reward_type == "distance_to_logs":
            # Reward based on distance to logs and penalty for collision
            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            log_trajectory = LogTrajectory.from_tensor(
                self.sim.expert_trajectory_tensor(),
                self.num_worlds,
                self.max_agent_count,
                backend=self.backend,
            )

            # Index log positions at current time steps
            log_traj_pos = []
            for i in range(self.num_worlds):
                log_traj_pos.append(
                    log_trajectory.pos_xy[i, :, world_time_steps[i], :]
                )
            log_traj_pos_tensor = torch.stack(log_traj_pos)

            agent_state = GlobalEgoState.from_tensor(
                self.sim.absolute_self_observation_tensor(),
                self.backend,
            )

            agent_pos = torch.stack(
                [agent_state.pos_x, agent_state.pos_y], dim=-1
            )

            # compute euclidean distance between agent and logs
            dist_to_logs = torch.norm(log_traj_pos_tensor - agent_pos, dim=-1)

            # add reward based on inverse distance to logs
            weighted_rewards += log_distance_weight * torch.exp(-dist_to_logs)

            return weighted_rewards

    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()
        not_done_worlds = ~self.get_dones().any(
            dim=1
        )  # Check if any agent in world is done
        self.world_time_steps[not_done_worlds] += 1

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
            or self.config.dynamics_model == "delta_local"
        ):
            if actions.dim() == 2:  # (num_worlds, max_agent_count)
                # Map action indices to action values if indices are provided
                actions = (
                    torch.nan_to_num(actions, nan=0).long().to(self.device)
                )
                action_value_tensor = self.action_keys_tensor[actions]

            elif actions.dim() == 3:
                if actions.shape[2] == 1:
                    actions = actions.squeeze(dim=2).to(self.device)
                    action_value_tensor = self.action_keys_tensor[actions]
                else:  # Assuming we are given the actual action values
                    action_value_tensor = actions.to(self.device)
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")

        else:
            action_value_tensor = actions.to(self.device)

        # Feed the action values to gpudrive
        self._copy_actions_to_simulator(action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provided actions to the simulator."""
        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            # Action space: (acceleration, steering, heading)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "delta_local":
            # Action space: (dx, dy, dyaw)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "state":
            # Following the StateAction struct in types.hpp
            # Need to provide:
            # (x, y, z, yaw, vel x, vel y, vel z, ang_vel_x, ang_vel_y, ang_vel_z)
            self.sim.action_tensor().to_torch()[:, :, :10].copy_(actions)
        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space based on dynamics model."""
        products = None

        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            products = product(self.dx, self.dy, self.dyaw)
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = self.config.head_tilt_actions.to(self.device)
            products = product(
                self.accel_actions, self.steer_actions, self.head_actions
            )
        elif self.config.dynamics_model == "state":
            self.x = self.config.x.to(self.device)
            self.y = self.config.y.to(self.device)
            self.yaw = self.config.yaw.to(self.device)
            self.vx = self.config.vx.to(self.device)
            self.vy = self.config.vy.to(self.device)

        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
        self.values_to_action_key = {}
        if products is not None:
            for action_idx, (action_1, action_2, action_3) in enumerate(
                products
            ):
                self.action_key_to_values[action_idx] = [
                    action_1.item(),
                    action_2.item(),
                    action_3.item(),
                ]
                self.values_to_action_key[
                    round(action_1.item(), 5),
                    round(action_2.item(), 5),
                    round(action_3.item(), 5),
                ] = action_idx

            self.action_keys_tensor = torch.tensor(
                [
                    self.action_key_to_values[key]
                    for key in sorted(self.action_key_to_values.keys())
                ]
            ).to(self.device)

            return Discrete(n=int(len(self.action_key_to_values)))
        else:
            return Discrete(n=1)

    def _set_continuous_action_space(self) -> None:
        """Configure the continuous action space."""
        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            action_1 = self.dx.clone().cpu().numpy()
            action_2 = self.dy.clone().cpu().numpy()
            action_3 = self.dyaw.clone().cpu().numpy()
        elif self.config.dynamics_model == "classic":
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_1 = self.steer_actions.clone().cpu().numpy()
            action_2 = self.accel_actions.clone().cpu().numpy()
            action_3 = self.head_actions.clone().cpu().numpy()
        else:
            raise ValueError(
                f"Continuous action space is currently not supported for dynamics_model: {self.config.dynamics_model}."
            )

        action_space = Tuple(
            (
                Box(action_1.min(), action_1.max(), shape=(1,)),
                Box(action_2.min(), action_2.max(), shape=(1,)),
                Box(action_3.min(), action_3.max(), shape=(1,)),
            )
        )
        return action_space

    def _get_ego_state(self, mask=None) -> torch.Tensor:
        """Get the ego state."""

        if not self.config.ego_state:
            return torch.Tensor().to(self.device)

        ego_state = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim.self_observation_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )
        if self.config.norm_obs:
            ego_state.normalize()

        if mask is None:
            if self.config.reward_type == "reward_conditioned":
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                        self.reward_weights_tensor[:, :, 0],
                        self.reward_weights_tensor[:, :, 1],
                        self.reward_weights_tensor[:, :, 2],
                    ]
                ).permute(1, 2, 0)

            else:
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                    ]
                ).permute(1, 2, 0)

        else:
            if self.config.reward_type == "reward_conditioned":
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                        self.reward_weights_tensor[mask][:, 0],
                        self.reward_weights_tensor[mask][:, 1],
                        self.reward_weights_tensor[mask][:, 2],
                    ]
                ).permute(1, 0)
            else:
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                    ]
                ).permute(1, 0)

    def _get_partner_obs(self, mask=None):
        """Get partner observations."""

        if not self.config.partner_obs:
            return torch.Tensor().to(self.device)

        partner_obs = PartnerObs.from_tensor(
            partner_obs_tensor=self.sim.partner_observations_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )

        if self.config.norm_obs:
            partner_obs.normalize()

        if mask is not None:
            return partner_obs.data.flatten(start_dim=1)
        else:
            return torch.concat(
                [
                    partner_obs.speed,
                    partner_obs.rel_pos_x,
                    partner_obs.rel_pos_y,
                    partner_obs.orientation,
                    partner_obs.vehicle_length,
                    partner_obs.vehicle_width,
                ],
                dim=-1,
            ).flatten(start_dim=2)

    def _get_road_map_obs(self, mask=None):
        """Get road map observations."""
        if not self.config.road_map_obs:
            return torch.Tensor().to(self.device)

        roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=self.sim.agent_roadmap_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )
        roadgraph.one_hot_encode_road_point_types()

        if self.config.norm_obs:
            roadgraph.normalize()

        if mask is not None:
            return torch.cat(
                [
                    roadgraph.data,
                    roadgraph.type,
                ],
                dim=-1,
            ).flatten(start_dim=1)
        else:
            return torch.cat(
                [
                    roadgraph.x.unsqueeze(-1),
                    roadgraph.y.unsqueeze(-1),
                    roadgraph.segment_length.unsqueeze(-1),
                    roadgraph.segment_width.unsqueeze(-1),
                    roadgraph.segment_height.unsqueeze(-1),
                    roadgraph.orientation.unsqueeze(-1),
                    roadgraph.type,
                ],
                dim=-1,
            ).flatten(start_dim=2)

    def _get_lidar_obs(self, mask=None):
        """Get lidar observations."""

        if not self.config.lidar_obs:
            return torch.Tensor().to(self.device)

        lidar = LidarObs.from_tensor(
            lidar_tensor=self.sim.lidar_tensor(),
            backend=self.backend,
            device=self.device,
        )

        if mask is not None:
            return [
                lidar.agent_samples[mask],
                lidar.road_edge_samples[mask],
                lidar.road_line_samples[mask],
            ]
        else:
            return torch.cat(
                [
                    lidar.agent_samples,
                    lidar.road_edge_samples,
                    lidar.road_line_samples,
                ],
                dim=-1,
            ).flatten(start_dim=2)

    def _get_bev_obs(self, mask=None):
        """Get BEV segmentation map observation.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, resolution, resolution, 1)
        """
        if not self.config.bev_obs:
            return torch.Tensor().to(self.device)

        bev = BevObs.from_tensor(
            bev_tensor=self.sim.bev_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        bev.one_hot_encode_bev_map()

        if mask is not None:
            return bev.bev_segmentation_map[mask].flatten(start_dim=1)
        else:
            return bev.bev_segmentation_map.flatten(start_dim=2)

    def _get_vbd_obs(self, mask=None):
        """
        Get ego-centric VBD trajectory observations for controlled agents using matrix operations.

        Args:
            mask: Optional mask to filter agents

        Returns:
            Tensor of ego-centric VBD trajectories
        """
        if not self.use_vbd or self.vbd_model is None:
            return torch.Tensor().to(self.device)

        # Get current agent positions and orientations
        agent_state = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=self.sim.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # Initialize output tensor
        traj_feature_dim = (
            self.vbd_trajectories.shape[2] * self.vbd_trajectories.shape[3]
        )

        if mask is not None:
            # Count valid agents for output tensor size
            valid_count = mask.sum().item()
            ego_vbd_trajectories = torch.zeros(
                (valid_count, traj_feature_dim), device=self.device
            )

            # Track which output index we're filling
            out_idx = 0

            # Process each world
            for w in range(self.num_worlds):
                # Get valid agent indices for this world
                world_mask = mask[w]
                agent_indices = torch.where(world_mask)[0]

                if len(agent_indices) == 0:
                    continue

                # Extract ego positions and yaws for these agents
                ego_pos_x = agent_state.pos_x[w, agent_indices]
                ego_pos_y = agent_state.pos_y[w, agent_indices]
                ego_yaw = agent_state.rotation_angle[w, agent_indices]

                # Process each agent in this world
                for i, agent_idx in enumerate(agent_indices):
                    # Get global trajectory for this agent
                    global_traj = self.vbd_trajectories[w, agent_idx]

                    # Create 2D rotation matrix for this agent
                    cos_yaw = torch.cos(ego_yaw[i])
                    sin_yaw = torch.sin(ego_yaw[i])
                    rotation_matrix = torch.tensor(
                        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
                        device=self.device,
                    )

                    # Transform positions using matrix multiplication
                    pos_xy = global_traj[:, :2]
                    ego_pos = torch.tensor(
                        [ego_pos_x[i], ego_pos_y[i]], device=self.device
                    ).reshape(1, 2)
                    translated_pos = (
                        pos_xy - ego_pos
                    )  # Broadcasting to subtract from all timesteps
                    rotated_pos = torch.matmul(
                        translated_pos, rotation_matrix.T
                    )

                    # Transform velocities (only rotation, no translation)
                    vel_xy = global_traj[:, 3:5]
                    rotated_vel = torch.matmul(vel_xy, rotation_matrix.T)

                    # Create transformed trajectory
                    transformed_traj = torch.zeros_like(global_traj)
                    transformed_traj[:, :2] = rotated_pos
                    transformed_traj[:, 2] = (
                        global_traj[:, 2] - ego_yaw[i]
                    )  # Adjust heading
                    transformed_traj[:, 3:5] = rotated_vel

                    # Flatten and add to output
                    ego_vbd_trajectories[out_idx] = transformed_traj.reshape(
                        -1
                    )
                    out_idx += 1

            if self.config.norm_obs:
                traj_len = self.vbd_trajectories.shape[2]
                ego_vbd_trajectories = self._normalize_vbd_obs(
                    ego_vbd_trajectories, traj_len
                )

            return ego_vbd_trajectories

        else:
            # Without mask, process all agents in all worlds
            ego_vbd_trajectories = torch.zeros(
                (self.num_worlds, self.max_agent_count, traj_feature_dim),
                device=self.device,
            )

            # Process each world
            for w in range(self.num_worlds):
                # Get controlled agent indices for this world
                valid_mask = self.cont_agent_mask[w]
                world_agent_indices = torch.where(valid_mask)[0]

                if len(world_agent_indices) == 0:
                    continue

                # Extract ego positions and yaws
                ego_pos_x = agent_state.pos_x[w]
                ego_pos_y = agent_state.pos_y[w]
                ego_yaw = agent_state.rotation_angle[w]

                # Process each agent in this world
                for agent_idx in world_agent_indices:
                    # Get global trajectory
                    global_traj = self.vbd_trajectories[w, agent_idx]

                    # Create 2D rotation matrix for this agent
                    cos_yaw = torch.cos(ego_yaw[agent_idx])
                    sin_yaw = torch.sin(ego_yaw[agent_idx])
                    rotation_matrix = torch.tensor(
                        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
                        device=self.device,
                    )

                    # Transform positions
                    pos_xy = global_traj[:, :2]
                    ego_pos = torch.tensor(
                        [ego_pos_x[agent_idx], ego_pos_y[agent_idx]],
                        device=self.device,
                    ).reshape(1, 2)
                    translated_pos = pos_xy - ego_pos
                    rotated_pos = torch.matmul(
                        translated_pos, rotation_matrix.T
                    )

                    # Transform velocities
                    vel_xy = global_traj[:, 3:5]
                    rotated_vel = torch.matmul(vel_xy, rotation_matrix.T)

                    # Create transformed trajectory
                    transformed_traj = torch.zeros_like(global_traj)
                    transformed_traj[:, :2] = rotated_pos
                    transformed_traj[:, 2] = (
                        global_traj[:, 2] - ego_yaw[agent_idx]
                    )
                    transformed_traj[:, 3:5] = rotated_vel

                    # Flatten and add to output
                    ego_vbd_trajectories[
                        w, agent_idx
                    ] = transformed_traj.reshape(-1)

            if self.config.norm_obs:
                traj_len = self.vbd_trajectories.shape[2]
                ego_vbd_trajectories = self._normalize_vbd_obs(
                    ego_vbd_trajectories, traj_len
                )

            return ego_vbd_trajectories

    def _normalize_vbd_obs(self, trajectories_flat, traj_len):
        """
        Normalize flattened VBD trajectory values to be between -1 and 1, with clipping.

        Args:
            trajectories_flat: Flattened tensor containing trajectory data
            traj_len: Number of trajectory steps

        Returns:
            Normalized flattened trajectories tensor
        """
        # Get original shape for proper reshaping
        original_shape = trajectories_flat.shape

        # Calculate feature dimension
        feature_dim = 5  # x, y, yaw, vel_x, vel_y

        # Reshape to separate the features
        if len(original_shape) == 2:  # (num_agents, flattened_features)
            traj_features = trajectories_flat.reshape(
                -1, traj_len, feature_dim
            )
        else:  # (num_worlds, max_agents, flattened_features)
            traj_features = trajectories_flat.reshape(
                original_shape[0], original_shape[1], traj_len, feature_dim
            )

        # Normalize each feature
        # x, y positions
        traj_features[..., 0] = normalize_min_max(
            tensor=traj_features[..., 0],
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )
        traj_features[..., 1] = normalize_min_max(
            tensor=traj_features[..., 1],
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )

        # Normalize yaw angle
        traj_features[..., 2] = (
            traj_features[..., 2] / constants.MAX_ORIENTATION_RAD
        )

        # Normalize velocities
        traj_features[..., 3] = traj_features[..., 3] / constants.MAX_SPEED
        traj_features[..., 4] = traj_features[..., 4] / constants.MAX_SPEED

        # Clip all values to the [-1, 1] range
        traj_features = torch.clamp(traj_features, min=-1.0, max=1.0)

        # Reshape back to original format
        return traj_features.reshape(original_shape)

    def get_obs(self, mask=None):
        """Get observation: Combine different types of environment information into a single tensor.
        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """
        # Base observations
        ego_states = self._get_ego_state(mask)
        partner_observations = self._get_partner_obs(mask)
        road_map_observations = self._get_road_map_obs(mask)

        if (
            self.use_vbd
            and self.vbd_model is not None
            and self.config.vbd_in_obs
        ):
            # Add ego-centric VBD trajectories
            vbd_observations = self._get_vbd_obs(mask)

            obs = torch.cat(
                (
                    ego_states,
                    partner_observations,
                    road_map_observations,
                    vbd_observations,
                ),
                dim=-1,
            )
        else:
            obs = torch.cat(
                (
                    ego_states,
                    partner_observations,
                    road_map_observations,
                ),
                dim=-1,
            )

        return obs

    def get_controlled_agents_mask(self):
        """Get the control mask. Shape: [num_worlds, max_agent_count]"""
        return (
            self.sim.controlled_state_tensor().to_torch().clone() == 1
        ).squeeze(axis=2)

    def advance_sim_with_log_playback(self, init_steps=0):
        """Advances the simulator by stepping the objects with the logged human trajectories.

        Args:
            init_steps (int): Number of warmup steps.
        """
        if init_steps >= self.config.episode_len:
            raise ValueError(
                "The length of the expert trajectory is 91,"
                f"so init_steps = {init_steps} should be < than 91."
            )

        self.init_frames = []

        self.log_playback_traj, _, _, _ = self.get_expert_actions()

        for time_step in range(init_steps):
            self.step_dynamics(
                actions=self.log_playback_traj[:, :, time_step, :]
            )

    def remove_agents_by_id(
        self, perc_to_rmv_per_scene, remove_controlled_agents=True
    ):
        """Delete random agents in scenarios.

        Args:
            perc_to_rmv_per_scene (float): Percentage of agents to remove per scene
            remove_controlled_agents (bool): If True, removes controlled agents. If False, removes uncontrolled agents
        """
        # Obtain agent ids
        agent_ids = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim.self_observation_tensor(),
            backend="torch",
            device=self.device,
        ).id

        # Choose the appropriate mask based on whether we're removing controlled or uncontrolled agents
        if remove_controlled_agents:
            agent_mask = self.cont_agent_mask
        else:
            # Create inverse mask for uncontrolled agents
            agent_mask = ~self.cont_agent_mask

        for env_idx in range(self.num_worlds):
            # Get all relevant agent IDs (controlled or uncontrolled) for the current environment
            scene_agent_ids = agent_ids[env_idx, :][agent_mask[env_idx]].long()

            if (
                scene_agent_ids.numel() > 0
            ):  # Ensure there are agents to sample
                # Determine the number of agents to sample (X% of the total agents)
                num_to_sample = max(
                    1, int(perc_to_rmv_per_scene * scene_agent_ids.size(0))
                )

                # Randomly sample agent IDs to remove using torch
                sampled_indices = torch.randperm(scene_agent_ids.size(0))[
                    :num_to_sample
                ]
                sampled_agent_ids = scene_agent_ids[sampled_indices]

                # Delete the sampled agents from the environment
                self.sim.deleteAgents({env_idx: sampled_agent_ids.tolist()})

        # Reset controlled agent mask and visualizer
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)

    def swap_data_batch(self, data_batch=None):
        """
        Swap the current data batch in the simulator with a new one
        and reinitialize dependent attributes.
        """

        if data_batch is None:  # Sample new data batch from the data loader
            self.data_batch = next(self.data_iterator)
        else:
            self.data_batch = data_batch

        # Validate that the number of worlds (envs) matches the batch size
        if len(self.data_batch) != self.num_worlds:
            raise ValueError(
                f"Data batch size ({len(self.data_batch)}) does not match "
                f"the expected number of worlds ({self.num_worlds})."
            )

        # Update the simulator with the new data
        self.sim.set_maps(self.data_batch)

        # Reinitialize the mask for controlled agents
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Generate VBD trajectories for the new batch if VBD is enabled
        if self.use_vbd and self.vbd_model is not None:
            self._generate_vbd_trajectories()

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)

    def _generate_vbd_trajectories(self):
        """Generate and store trajectory predictions for all scenes using VBD model."""
        if not self.use_vbd or self.vbd_model is None:
            return

        _ = self.reset()

        # Generate sample batch using the limited mask
        sample_batch = self._generate_sample_batch(init_steps=self.init_steps)

        # VBD model prediction
        predictions = self.vbd_model.sample_denoiser(sample_batch)
        vbd_trajectories = (
            predictions["denoised_trajs"].to(self.device).numpy()
        )
        agent_indices = sample_batch["agents_id"]

        self.vbd_trajectories.zero_()
        # Process each world separately
        for world_idx in range(self.num_worlds):
            world_agent_indices = agent_indices[world_idx]

            # Filter out negative indices (they're our padding)
            valid_mask = (
                world_agent_indices >= 0
            )  # Boolean mask of valid indices
            valid_agent_indices = world_agent_indices[
                valid_mask
            ]  # Filtered tensor

            if len(valid_agent_indices) > 0:
                # Update vbd_trajectories(x, y, yaw, vel_x, vel_y) for this world's agents
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, :2
                ] = torch.Tensor(
                    vbd_trajectories[
                        world_idx, : len(valid_agent_indices), :, :2
                    ]
                )
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, :2
                ] -= self.sim.world_means_tensor().to_torch()[
                    world_idx, :2
                ]  # subtract mean
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, 2
                ] = torch.Tensor(
                    vbd_trajectories[
                        world_idx, : len(valid_agent_indices), :, 2
                    ]
                )
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, 3:
                ] = torch.Tensor(
                    vbd_trajectories[
                        world_idx, : len(valid_agent_indices), :, 3:5
                    ]
                )

    def get_expert_actions(self):
        """Get expert actions for the full trajectories across worlds.

        Returns:
            expert_actions: Inferred or logged actions for the agents.
            expert_speeds: Speeds from the logged trajectories.
            expert_positions: Positions from the logged trajectories.
            expert_yaws: Heading from the logged trajectories.
        """

        log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
        )

        if self.config.dynamics_model == "delta_local":
            inferred_actions = log_trajectory.inferred_actions[..., :3]
            inferred_actions[..., 0] = torch.clamp(
                inferred_actions[..., 0], -6, 6
            )
            inferred_actions[..., 1] = torch.clamp(
                inferred_actions[..., 1], -6, 6
            )
            inferred_actions[..., 2] = torch.clamp(
                inferred_actions[..., 2], -torch.pi, torch.pi
            )
        elif self.config.dynamics_model == "state":
            # Extract (x, y, yaw, velocity x, velocity y)
            inferred_actions = torch.cat(
                (
                    log_trajectory.pos_xy,
                    torch.ones(
                        (*log_trajectory.pos_xy.shape[:-1], 1),
                        device=self.device,
                    ),
                    log_trajectory.yaw,
                    log_trajectory.vel_xy,
                    torch.zeros(
                        (*log_trajectory.pos_xy.shape[:-1], 4),
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            inferred_actions = log_trajectory.inferred_actions[..., :3]
            inferred_actions[..., 0] = torch.clamp(
                inferred_actions[..., 0], -6, 6
            )
            inferred_actions[..., 1] = torch.clamp(
                inferred_actions[..., 1], -0.3, 0.3
            )

        return (
            inferred_actions,
            log_trajectory.pos_xy,
            log_trajectory.vel_xy,
            log_trajectory.yaw,
        )

    def get_env_filenames(self):
        """Obtain the tfrecord filename for each world, mapping world indices to map names."""

        map_name_integers = self.sim.map_name_tensor().to_torch()
        filenames = {}
        # Iterate through the number of worlds
        for i in range(self.num_worlds):
            tensor = map_name_integers[i]
            # Convert ints to characters, ignoring zeros
            map_name = "".join([chr(i) for i in tensor.tolist() if i != 0])
            filenames[i] = map_name

        return filenames

    def get_scenario_ids(self):
        """Obtain the scenario ID for each world."""
        scenario_id_integers = self.sim.scenario_id_tensor().to_torch()
        scenario_ids = {}

        # Iterate through the number of worlds
        for i in range(self.num_worlds):
            tensor = scenario_id_integers[i]
            # Convert ints to characters, ignoring zeros
            scenario_id = "".join([chr(i) for i in tensor.tolist() if i != 0])
            scenario_ids[i] = scenario_id

        return scenario_ids


if __name__ == "__main__":

    env_config = EnvConfig(
        dynamics_model="delta_local",
    )
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/examples",
        batch_size=2,
        dataset_size=100,
        sample_with_replacement=True,
        shuffle=False,
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=64,  # Number of agents to control
        device="cpu",
    )

    control_mask = env.cont_agent_mask

    # Rollout
    obs = env.reset()

    sim_frames = []
    agent_obs_frames = []

    expert_actions, _, _, _ = env.get_expert_actions()

    env_idx = 0

    for t in range(10):
        print(f"Step: {t}")

        # Step the environment
        expert_actions, _, _, _ = env.get_expert_actions()
        env.step_dynamics(expert_actions[:, :, t, :])

        highlight_agent = torch.where(env.cont_agent_mask[env_idx, :])[0][
            -1
        ].item()

        # Make video
        sim_states = env.vis.plot_simulator_state(
            env_indices=[env_idx],
            zoom_radius=50,
            time_steps=[t],
            center_agent_indices=[highlight_agent],
        )

        agent_obs = env.vis.plot_agent_observation(
            env_idx=env_idx,
            agent_idx=highlight_agent,
            figsize=(10, 10),
        )

        sim_frames.append(img_from_fig(sim_states[0]))
        agent_obs_frames.append(img_from_fig(agent_obs))

        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()
        info = env.get_infos()

        if done[0, highlight_agent].bool():
            break

    env.close()

    media.write_video(
        "sim_video.gif", np.array(sim_frames), fps=10, codec="gif"
    )
    media.write_video(
        "obs_video.gif", np.array(agent_obs_frames), fps=10, codec="gif"
    )
