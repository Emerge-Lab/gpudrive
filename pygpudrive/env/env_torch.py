"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from itertools import product

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.base_env import GPUDriveGymEnv


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
    ):
        # Initialization of environment configurations
        self.config = config
        self.num_worlds = scene_config.num_scenes
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config

        # Environment parameter setup
        params = self._setup_environment_parameters()

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

        # Rendering setup
        self.visualizer = self._setup_rendering()

    def reset(self):
        """Reset the worlds and return the initial observations."""
        self.sim.reset(list(range(self.num_worlds)))
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

        assert actions.shape == (
            self.num_worlds,
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
            ego_states = torch.Tensor().to(self.device)

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
                road_map_observations = road_map_observations.flatten(
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
        return (self.sim.controlled_state_tensor().to_torch() == 1).squeeze(
            axis=2
        )

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= self.config.max_speed
        state[:, :, 1] /= self.config.max_veh_len
        state[:, :, 2] /= self.config.max_veh_width

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

        # Object type
        shifted_type_obs = obs[:, :, :, 6] - 6
        one_hot_object_type = torch.nn.functional.one_hot(
            torch.where(
                condition=shifted_type_obs >= 0,
                input=shifted_type_obs,
                other=0,
            ).long(),
            num_classes=4,
        )
        # Concatenate the one-hot encoding with the rest of the features
        obs = torch.concat((obs[:, :, :, :6], one_hot_object_type), dim=-1)

        return obs.flatten(start_dim=2)

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

        # TODO: Road scale (width and height)
        obs[:, :, :, 3] /= self.config.max_road_scale
        # obs[:, :, :, 4] seems already scaled

        # Road point orientation
        obs[:, :, :, 5] /= self.config.max_orientation_rad

        # Road types: one-hot encode them
        one_hot_road_type = torch.nn.functional.one_hot(
            obs[:, :, :, 6].long(), num_classes=7
        )

        # Concatenate the one-hot encoding with the rest of the features (exclude index 3 and 4)
        obs = torch.cat((obs[:, :, :, :6], one_hot_road_type), dim=-1)

        return obs.flatten(start_dim=2)


if __name__ == "__main__":

    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_NUM_OBJECTS = 128
    NUM_WORLDS = 50

    env_config = EnvConfig()
    render_config = RenderConfig()
    scene_config = SceneConfig(path="data", num_scenes=NUM_WORLDS)

    # MAKE ENV
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
        device="cuda",
        render_config=render_config,
    )

    # RUN
    obs = env.reset()

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

        obs = env.get_obs()

        reward = env.get_rewards()
        done = env.get_dones()

    # import imageio
    # imageio.mimsave("world1.gif", frames_1)

    # run.finish()
    env.visualizer.destroy()
