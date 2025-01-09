"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
from itertools import product
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.base_env import GPUDriveGymEnv

from pygpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
    LidarObs,
)
from pygpudrive.datatypes.trajectory import LogTrajectory
from pygpudrive.datatypes.roadgraph import LocalRoadGraphPoints
from pygpudrive.datatypes.info import Info

from pygpudrive.visualize.core import MatplotlibVisualizer
from pygpudrive.env.dataset import SceneDataLoader


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

        # Environment parameter setup
        params = self._setup_environment_parameters()

        # Initialize the iterator once
        self.data_iterator = iter(self.data_loader)

        # Get the initial data batch
        self.data_batch = next(self.data_iterator)

        # Initialize simulator
        self.sim = self._initialize_simulator(params, self.data_batch)

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.get_obs().shape[-1],)
        )
        self._setup_action_space(action_type)
        self.info_dim = 5  # Number of info features
        self.episode_len = self.config.episode_len

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

    def reset(self):
        """Reset the worlds and return the initial observations."""
        self.sim.reset(list(range(self.num_worlds)))
        return self.get_obs()

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
        collision_weight=-0.005,
        goal_achieved_weight=1.0,
        off_road_weight=-0.005,
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
        if self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

        elif self.config.reward_type == "weighted_combination":
            # Return the weighted combination of the reward components
            info_tensor = self.sim.info_tensor().to_torch().clone()
            off_road = info_tensor[:, :, 0].to(torch.float)

            # True if the vehicle is in collision with another road object
            # (i.e. a cyclist or pedestrian)
            collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
            goal_achieved = info_tensor[:, :, 3].to(torch.float)

            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            return weighted_rewards

        elif self.config.reward_type == "distance_to_logs":
            # Reward based on distance to logs and penalty for collision

            # Return the weighted combination of the reward components
            info_tensor = self.sim.info_tensor().to_torch().clone()
            off_road = info_tensor[:, :, 0].to(torch.float)

            # True if the vehicle collided with another road object
            # (i.e. a cyclist or pedestrian)
            collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
            goal_achieved = info_tensor[:, :, 3].to(torch.float)

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
                elif (
                    actions.shape[2] == 3
                ):  # Assuming we are given the actual action values
                    # (acceleration, steering, heading)
                    action_value_tensor = actions.to(self.device)
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")

        else:
            action_value_tensor = actions.to(self.device)

        # Feed the action values to gpudrive
        self._copy_actions_to_simulator(action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provived actions to the simulator."""
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
            # Need to provide: (x, y, z, yaw, velocity x, vel y, vel z, ang_vel_x, ang_vel_y, ang_vel_z)
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

    def _get_ego_state(self) -> torch.Tensor:
        """Get the ego state.
        Returns:
            Shape: (num_worlds, max_agents, num_features)
        """
        if self.config.ego_state:
            ego_state = LocalEgoState.from_tensor(
                self_obs_tensor=self.sim.self_observation_tensor(),
                backend=self.backend,
            )
            if self.config.norm_obs:
                ego_state.normalize()

            return (
                torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                    ]
                )
                .permute(1, 2, 0)
                .to(self.device)
            )
        else:
            return torch.Tensor().to(self.device)

    def _get_partner_obs(self):
        """Get partner observations."""
        if self.config.partner_obs:
            partner_obs = PartnerObs.from_tensor(
                partner_obs_tensor=self.sim.partner_observations_tensor(),
                backend=self.backend,
            )

            if self.config.norm_obs:
                partner_obs.normalize()
                partner_obs.one_hot_encode_agent_types()

            return (
                torch.concat(
                    [
                        partner_obs.speed,
                        partner_obs.rel_pos_x,
                        partner_obs.rel_pos_y,
                        partner_obs.orientation,
                        partner_obs.vehicle_length,
                        partner_obs.vehicle_width,
                        # partner_obs.agent_type,
                    ],
                    dim=-1,
                )
                .flatten(start_dim=2)
                .to(self.device)
            )

        else:
            return torch.Tensor().to(self.device)

    def _get_road_map_obs(self):
        """Get road map observations."""
        if self.config.road_map_obs:
            roadgraph = LocalRoadGraphPoints.from_tensor(
                local_roadgraph_tensor=self.sim.agent_roadmap_tensor(),
                backend=self.backend,
            )

            if self.config.norm_obs:
                roadgraph.normalize()
                roadgraph.one_hot_encode_road_point_types()

            return (
                torch.cat(
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
                )
                .flatten(start_dim=2)
                .to(self.device)
            )

        else:
            return torch.Tensor().to(self.device)

    def _get_lidar_obs(self):
        """Get lidar observations."""
        if self.config.lidar_obs:
            lidar = LidarObs.from_tensor(
                lidar_tensor=self.sim.lidar_tensor(),
                backend=self.backend,
            )

            return (
                torch.cat(
                    [
                        lidar.agent_samples,
                        lidar.road_edge_samples,
                        lidar.road_line_samples,
                    ],
                    dim=-1,
                )
                .flatten(start_dim=2)
                .to(self.device)
            )
        else:
            return torch.Tensor().to(self.device)

    def get_obs(self):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        ego_states = self._get_ego_state()

        partner_observations = self._get_partner_obs()

        road_map_observations = self._get_road_map_obs()

        lidar_obs = self._get_lidar_obs()

        obs_filtered = torch.cat(
            (
                ego_states,
                partner_observations,
                road_map_observations,
                lidar_obs,
            ),
            dim=-1,
        )

        return obs_filtered

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (
            self.sim.controlled_state_tensor().to_torch().clone() == 1
        ).squeeze(axis=2)

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

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)

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


if __name__ == "__main__":

    from pygpudrive.visualize.utils import img_from_fig
    import mediapy as media

    env_config = EnvConfig(dynamics_model="delta_local")
    render_config = RenderConfig()
    data_config = SceneConfig(batch_size=2, dataset_size=1000)

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/training",
        batch_size=data_config.batch_size,
        dataset_size=data_config.dataset_size,
        sample_with_replacement=True,
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=128,  # Number of agents to control
        device="cpu",
    )

    print(f"dataset: {env.data_batch}")

    print(
        f"controlled agents mask [before reset]: {env.cont_agent_mask.sum()}"
    )

    # Rollout
    obs = env.reset()

    print(f"controlled agents mask: {env.cont_agent_mask.sum()}")

    sim_frames = []
    agent_obs_frames = []

    # env.swap_data_batch()
    # env.reset()

    expert_actions, _, _, _ = env.get_expert_actions()

    env_idx = 0

    for t in range(10):
        print(f"Step: {t}")

        # Step the environment
        env.step_dynamics(expert_actions[:, :, t, :])

        # if (t + 1) % 2 == 0:
        # env.swap_data_batch()
        # env.reset()
        #     print(f"dataset: {env.data_batch}")

        # sim_state[0].savefig(f"sim_state.png")   # Save the figure to a file
        # agent_obs_fig.savefig(f"agent_obs.png")  # Save the figure to a file

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

        sim_states[0].savefig(f"sim_state.png")  # Save the figure to a file
        agent_obs.savefig(f"agent_obs.png")  # Save the figure to a file

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
