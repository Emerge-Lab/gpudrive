"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
import argparse
import gpudrive
import imageio
from itertools import product

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.base_env import GPUDriveGymEnv
import logging

class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
            self,
            config,
            scene_config,
            max_cont_agents,
            device="cuda",
            action_type="discrete",
            render_config: RenderConfig = RenderConfig(),
            useDeltaModel: bool = False,
    ):
        # Initialization of environment configurations
        self.config = config
        self.num_worlds = scene_config.num_scenes
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config
        self.useDeltaModel = useDeltaModel

        # Environment parameter setup
        params = self._setup_environment_parameters()
        params.useDeltaModel = self.useDeltaModel
        params.useWayMaxModel = False if self.useDeltaModel else True

        # Initialize simulator with parameters
        self.sim = self._initialize_simulator(params, scene_config)
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
        self.episode_len = self.config.episode_length

        # Rendering setup
        self.visualizer = self._setup_rendering()

    def reset(self):
        """Reset the worlds and return the initial observations."""
        for sim_idx in range(self.num_worlds):
            self.sim.reset(sim_idx)
        self.sim.step()  # We require one step to trigger the reset
        return self.get_obs()

    def get_dones(self):
        return self.sim.done_tensor().to_torch().squeeze(dim=2).to(torch.float)

    def get_infos(self):
        return (
            self.sim.info_tensor()
                .to_torch()
                .squeeze(dim=2)
                .to(torch.float)
                .to(self.device)
        )

    def get_rewards(self):
        return self.sim.reward_tensor().to_torch().squeeze(dim=2)

    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        if actions.dim() == 2:  # (num_worlds, max_agent_count)
            # Map action indices to action values if indices are provided
            actions = torch.nan_to_num(actions, nan=0).long().to(self.device)
            action_value_tensor = self.action_keys_tensor[actions]

        elif actions.dim() == 3:
            if actions.shape[2] == 1:
                actions = actions.squeeze(dim=2).to(self.device)
                action_value_tensor = self.action_keys_tensor[actions]
            elif (
                    actions.shape[2] == 3
            ):  # Assuming we are given the actual action values (acceleration, steering, heading)
                action_value_tensor = actions.to(self.device)
        else:
            raise ValueError(f"Invalid action shape: {actions.shape}")
        # Feed the actual action values to gpudrive
        if self.useDeltaModel:
            # print("DELTAACTION ", action_value_tensor[0, 5])
            self.sim.delta_action_tensor().to_torch().copy_(action_value_tensor)
        else:
            self.sim.action_tensor().to_torch().copy_(action_value_tensor)

    def _set_discrete_action_space(self) -> None:
        """Configure a discrete joint action space.
        action = (acceleration, steering angle, heading angle)
        """
        self.accel_actions = self.config.accel_actions.to(self.device)
        self.steer_actions = self.config.steer_actions.to(self.device)
        self.head_actions = torch.tensor([0], device=self.device)

        # Map action indices -> action values and vice versa
        self.action_key_to_values = {}
        self.values_to_action_key = {}

        for action_idx, (accel, steer, head) in enumerate(
                product(self.accel_actions, self.steer_actions, self.head_actions)
        ):
            self.action_key_to_values[action_idx] = [
                accel.item(),
                steer.item(),
                head.item(),
            ]

            self.values_to_action_key[
                round(accel.item(), 3),
                round(steer.item(), 3),
                round(head.item(), 3),
            ] = action_idx

        self.action_keys_tensor = torch.tensor(
            [
                self.action_key_to_values[key]
                for key in sorted(self.action_key_to_values.keys())
            ]
        ).to(self.device)

        self.value_keys_tensor = torch.tensor(
            [
                self.values_to_action_key[key]
                for key in sorted(self.values_to_action_key.keys())
            ]
        ).to(self.device)

        return Discrete(n=int(len(self.action_key_to_values)))

    def _set_discrete_delta_action_space(self) -> None:
        """Configure a discrete joint delta action space.
        action = (dx, dy, dyaw)
        """
        print('Starting discrete delta action space...')
        self.dx = self.config.dx.to(self.device)
        self.dy = self.config.dy.to(self.device)
        self.dyaw = self.config.dyaw.to(self.device)

        # Map action indices -> action values and vice versa
        self.action_key_to_values = {}
        self.values_to_action_key = {}

        for action_idx, (dx, dy, dyaw) in enumerate(
                product(self.dx, self.dy, self.dyaw)
        ):
            self.action_key_to_values[action_idx] = [
                dx.item(),
                dy.item(),
                dyaw.item(),
            ]

            self.values_to_action_key[
                round(dx.item(), 3),
                round(dy.item(), 3),
                round(dyaw.item(), 3),
            ] = action_idx

        self.action_keys_tensor = torch.tensor(
            [
                self.action_key_to_values[key]
                for key in sorted(self.action_key_to_values.keys())
            ]
        ).to(self.device)
        self.value_keys_tensor = torch.tensor(
            [
                self.values_to_action_key[key]
                for key in sorted(self.values_to_action_key.keys())
            ]
        ).to(self.device)

        print('Ending discrete delta action space...')
        return Discrete(n=int(len(self.action_key_to_values)))

    def get_obs(self):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        # EGO STATE
        if self.config.ego_state:
            ego_states_unprocessed = (
                self.sim.self_observation_tensor().to_torch()
            )
            if self.config.norm_obs:
                ego_states = self.normalize_ego_state(ego_states_unprocessed)
            else:
                ego_states = ego_states_unprocessed
        else:
            ego_states = torch.Tensor().to(self.device)
        # abs_obs = self.sim.absolute_self_observation_tensor().to_torch()[0, 5, :2]
        # print('EGO Speed Unprocesssed',self.config.norm_obs, ego_states_unprocessed[0, 5, 0])

        # print('EGO PROCESSED', ego_states[0, 5, 3:5])
        # PARTNER OBSERVATIONS
        if self.config.partner_obs:
            partner_observations = (
                self.sim.partner_observations_tensor().to_torch()
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
            partner_observations = torch.Tensor().to(self.device)

        # ROAD MAP OBSERVATIONS
        if self.config.road_map_obs:

            road_map_observations_unprocessed = (
                self.sim.agent_roadmap_tensor().to_torch()
            )

            if self.config.norm_obs:
                road_map_observations = self.normalize_and_flatten_map_obs(
                    road_map_observations_unprocessed
                )
            else:
                road_map_observations = road_map_observations_unprocessed.flatten(
                    start_dim=2
                )
        else:
            road_map_observations = torch.Tensor().to(self.device)

        # Combine the observations
        obs_filtered = torch.cat(
            (
                ego_states,
                partner_observations,
                road_map_observations,
            ),
            dim=-1,
        )
        return obs_filtered

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (self.sim.controlled_state_tensor().to_torch() == 2).squeeze(
            axis=2
        )

    def get_expert_actions(self, debug_world_idx=None, debug_veh_idx=None):
        """Get expert actions for the full trajectories across worlds."""
        expert_traj = self.sim.expert_trajectory_tensor().to_torch()
        positions = expert_traj[:, :, :2 * self.episode_len].view(self.num_worlds,
                                                                  self.max_agent_count,
                                                                  self.episode_len, -1)

        velocity = expert_traj[:, :, 2 * self.episode_len:4 * self.episode_len].view(self.num_worlds,
                                                                                     self.max_agent_count,
                                                                                     self.episode_len, -1)
        if self.useDeltaModel:
            inferred_expert_actions = expert_traj[:, :, -3 * self.episode_len:].view(self.num_worlds,
                                                                                     self.max_agent_count,
                                                                                     self.episode_len, -1)
            inferred_expert_actions[..., 0] = torch.clamp(inferred_expert_actions[..., 0], -6, 6)
            inferred_expert_actions[..., 1] = torch.clamp(inferred_expert_actions[..., 1], -6, 6)
            inferred_expert_actions[..., 2] = torch.clamp(inferred_expert_actions[..., 2], -3.14, 3.14)
        else:
            inferred_expert_actions = expert_traj[:, :, -6 * self.episode_len:-3 * self.episode_len].view(
                self.num_worlds,
                self.max_agent_count,
                self.episode_len, -1)
            inferred_expert_actions[..., 0] = torch.clamp(inferred_expert_actions[..., 0], -6, 6)
            inferred_expert_actions[..., 1] = torch.clamp(inferred_expert_actions[..., 1], -0.3, 0.3)
        velo2speed = None
        debug_positions = None
        if debug_world_idx is not None and debug_veh_idx is not None:
            velo2speed = torch.norm(velocity[debug_world_idx, debug_veh_idx], dim=-1) / self.config.max_speed
            positions[..., 0] = self.normalize_tensor(
                positions[..., 0],
                self.config.min_rel_goal_coord,
                self.config.max_rel_goal_coord,
            )
            positions[..., 1] = self.normalize_tensor(
                positions[..., 1],
                self.config.min_rel_goal_coord,
                self.config.max_rel_goal_coord,
            )
            debug_positions = positions[debug_world_idx, debug_veh_idx]
        # print(f'Expert Trajectory 5 one speed {torch.norm(velocity[0, 5], dim=-1) / self.config.max_speed}, sum of speed : {torch.norm(velocity[0, 5], dim=-1).sum() / 100}')
        return inferred_expert_actions, velo2speed, debug_positions

    def normalize_ego_state(self, state):
        """Normalize ego state features."""
        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= self.config.max_speed
        state[:, :, 1] /= self.config.max_veh_len
        state[:, :, 2] /= self.config.max_veh_width

        # print(f'Check normalize after {state[0, 5, 0]}')
        # Relative goal coordinates
        state[:, :, 3] = self.normalize_tensor(
            state[:, :, 3],
            self.config.min_rel_goal_coord,
            self.config.max_rel_goal_coord,
        )
        state[:, :, 4] = self.normalize_tensor(
            state[:, :, 4],
            self.config.min_rel_goal_coord,
            self.config.max_rel_goal_coord,
        )

        # Uncommment this to exclude the collision state
        # (1 if vehicle is in collision, 1 otherwise)
        # state = state[:, :, :5]

        return state

    def normalize_and_flatten_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: torch.Tensor of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        """

        # TODO: Fix (there should not be nans in the obs)
        obs = torch.nan_to_num(obs, nan=0)

        # Speed
        obs[:, :, :, 0] /= self.config.max_speed

        # Relative position
        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            self.config.min_rel_agent_pos,
            self.config.max_rel_agent_pos,
        )
        obs[:, :, :, 2] = self.normalize_tensor(
            obs[:, :, :, 2],
            self.config.min_rel_agent_pos,
            self.config.max_rel_agent_pos,
        )

        # Orientation (heading)
        obs[:, :, :, 3] /= self.config.max_orientation_rad

        # Vehicle length and width
        obs[:, :, :, 4] /= self.config.max_veh_len
        obs[:, :, :, 5] /= self.config.max_veh_width

        # One-hot encode the type of the other visible objects
        one_hot_encoded_object_types = self.one_hot_encode_object_type(obs[:, :, :, 6])

        # Concat the one-hot encoding with the rest of the features
        obs = torch.concat((obs[:, :, :, :6], one_hot_encoded_object_types), dim=-1)

        return obs.flatten(start_dim=2)

    def one_hot_encode_roadpoints(self, roadmap_type_tensor):
        roadmap_type_tensor = torch.nan_to_num(roadmap_type_tensor, nan=0)
        # Set garbage object types to zero
        road_types = torch.where(
            (roadmap_type_tensor < self.MIN_OBJ_ENTITY_ENUM) | (roadmap_type_tensor >= self.ROAD_MAP_OBJECT_TYPES),
            0.0,
            roadmap_type_tensor,
            ).int()

        return torch.nn.functional.one_hot(
            road_types.long(), num_classes=self.ROAD_MAP_OBJECT_TYPES,
        )

    def one_hot_encode_object_type(self, object_type_tensor):
        """One-hot encode the object type."""

        VEHICLE = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Vehicle]
        PEDESTRIAN = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Pedestrian]
        CYCLIST = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Cyclist]
        PADDING = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType._None]

        # Set garbage object elements to zero
        object_types = torch.where(
            (object_type_tensor < self.MIN_OBJ_ENTITY_ENUM) | (object_type_tensor > self.MAX_OBJ_ENTITY_ENUM),
            0.0,
            object_type_tensor,
            ).int()

        one_hot_object_type = torch.nn.functional.one_hot(
            torch.where(
                condition=(object_types == VEHICLE) | (object_types == PEDESTRIAN) | (
                        object_types == CYCLIST) | object_types == PADDING,
                input=object_types,
                other=0,
            ).long(),
            num_classes=self.ROAD_OBJECT_TYPES,
        )
        return one_hot_object_type

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""

        # Road point coordinates
        obs[:, :, :, 0] = self.normalize_tensor(
            obs[:, :, :, 0],
            self.config.min_rm_coord,
            self.config.max_rm_coord,
        )

        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            self.config.min_rm_coord,
            self.config.max_rm_coord,
        )

        # Road line segment length
        obs[:, :, :, 2] /= self.config.max_road_line_segmment_len

        # Road scale (width and height)
        obs[:, :, :, 3] /= self.config.max_road_scale
        # obs[:, :, :, 4] seems already scaled

        # Road point orientation
        obs[:, :, :, 5] /= self.config.max_orientation_rad

        # Road types: one-hot encode them
        one_hot_road_types = self.one_hot_encode_roadpoints(obs[:, :, :, 6])

        # Concatenate the one-hot encoding with the rest of the features
        obs = torch.cat((obs[:, :, :, :6], one_hot_road_types), dim=-1)

        return obs.flatten(start_dim=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_delta_model', action='store_true', help='Use delta model to infer expert actions')
    args = parser.parse_args()

    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_NUM_OBJECTS = 128
    NUM_WORLDS = 30

    env_config = EnvConfig()
    render_config = RenderConfig()
    scene_config = SceneConfig("data", NUM_WORLDS)

    # MAKE ENV
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
        device="cpu",
        render_config=render_config,
        useDeltaModel=args.use_delta_model,
    )

    # RUN
    obs = env.reset()
    frames = []

    for _ in range(TOTAL_STEPS):
        # Take a random actions
        rand_action = torch.Tensor(
            [
                [
                    env.action_space.sample()
                    for _ in range(MAX_NUM_OBJECTS * NUM_WORLDS)
                ]
            ]
        ).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)

        # Step the environment
        env.step_dynamics(rand_action)

        frames.append(env.render())

        obs = env.get_obs()

        reward = env.get_rewards()
        done = env.get_dones()

    # import imageio
    imageio.mimsave("world1.gif", np.array(frames))

    # run.finish()
    env.visualizer.destroy()
