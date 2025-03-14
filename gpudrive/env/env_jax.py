from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
import jax
import jax.numpy as jnp

from itertools import product
import mediapy
import gymnasium

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.base_env import GPUDriveGymEnv

from gpudrive.visualize.core import MatplotlibVisualizer
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env import constants
from gpudrive.env.config import EnvConfig, RenderConfig, SceneConfig


class GPUDriveJaxEnv(GPUDriveGymEnv):
    """Jax Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        data_loader,
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
        backend="jax",
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

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(self.get_obs().shape[-1],)
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
        """Get dones for all agents."""
        return self.sim.done_tensor().to_jax().squeeze(axis=2)

    def get_infos(self):
        """Get info for all agents."""
        return self.sim.info_tensor().to_jax()

    def get_rewards(
        self,
        off_road_weight=0.75,
        collision_weight=0.75,
        goal_achieved_weight=1.0,
    ):
        """Get rewards for all agents."""
        if self.config.reward_type == "sparse_on_goal_achieved":
            self.sim.reward_tensor().to_jax().squeeze(axis=2)

        elif self.config.reward_type == "weighted_combination":
            infos = self.get_infos()

            # Indicators
            off_road = infos[:, :, 0]
            agent_collision = infos[:, :, 1:3].sum(axis=2)
            goal_achieved = infos[:, :, 3]

            # Weighted combination
            rewards = (
                goal_achieved_weight * goal_achieved
                - collision_weight * agent_collision
                - off_road_weight * off_road
            )

            return rewards

    def step_dynamics(self, actions):
        """Step the simulator."""
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        if self.config.dynamics_model in {"classic", "bicycle", "delta_local"}:
            if actions.ndim == 2:  # (num_worlds, max_agent_count)
                # Map action indices to action values if indices are provided
                actions = jnp.nan_to_num(actions, nan=0).astype(jnp.int32)
                action_value_tensor = self.action_keys_tensor[actions]

            elif actions.ndim == 3:
                if actions.shape[2] == 1:
                    actions = actions.squeeze(axis=2)
                    action_value_tensor = self.action_keys_tensor[actions]
                elif (
                    actions.shape[2] == 3
                ):  # Assuming actual action values are given
                    action_value_tensor = actions
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")
        else:
            action_value_tensor = actions

        # Feed the action values to gpudrive
        self._copy_actions_to_simulator(action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provided actions to the simulator."""

        # Convert to torch Tensor (tmp solution)
        actions = torch.from_numpy(np.array(actions))

        if self.config.dynamics_model in {"classic", "bicycle", "delta_local"}:
            # Action space: (acceleration, steering, heading) or (dx, dy, dyaw)
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

    def _get_ego_state(self):
        """Get the ego state."""
        if self.config.ego_state:
            ego_states = self.sim.self_observation_tensor().to_jax()
            if self.config.norm_obs:
                ego_states = self.process_ego_state(ego_states)
        else:
            ego_states = jnp.array()
        return ego_states

    def _get_partner_obs(self):
        """Get the partner observation."""
        if self.config.partner_obs:
            partner_observations = (
                self.sim.partner_observations_tensor().to_jax()
            )
            if self.config.norm_obs:
                partner_observations = self.process_partner_obs(
                    partner_observations
                )
            else:
                partner_observations = partner_observations.flatten(
                    start_dim=2
                )
        else:
            partner_observations = jnp.array()
        return partner_observations

    def _get_road_map_obs(self):
        """Get the road map observation."""
        if self.config.road_map_obs:
            road_map_observations = self.sim.agent_roadmap_tensor().to_jax()

            if self.config.norm_obs:
                road_map_observations = self.process_roadgraph(
                    road_map_observations
                )
            else:
                road_map_observations = road_map_observations.flatten(
                    start_dim=2
                )
        else:
            road_map_observations = jnp.array()
        return road_map_observations

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (self.sim.controlled_state_tensor().to_jax() == 1).squeeze(
            axis=2
        )

    def get_obs(self):
        """Get observation: Aggregate multi-modal environment information into
            a single flattened tensor. All information is in the shape of
            (num_worlds, max_agent_count, num_features).

        Returns:
            jnp.array: (num_worlds, max_agent_count, num_features)
        """

        ego_states = self._get_ego_state()
        partner_observations = self._get_partner_obs()
        road_map_observations = self._get_road_map_obs()

        obs_filtered = jnp.concatenate(
            (
                ego_states,
                partner_observations,
                road_map_observations,
            ),
            axis=-1,
        )
        return obs_filtered

    def process_ego_state(self, state):
        """Normalize ego state features."""
        indices = jnp.array([0, 1, 2, 4, 5, 6])

        # Speed, vehicle length, vehicle width
        state = state.at[:, :, 0].divide(constants.MAX_SPEED)
        state = state.at[:, :, 1].divide(constants.MAX_VEH_LEN)
        state = state.at[:, :, 2].divide(constants.MAX_VEH_WIDTH)

        # Skip vehicle height (3)

        # Relative goal coordinates
        state = state.at[:, :, 4].set(
            self.normalize_tensor(
                state[:, :, 4],
                constants.MIN_REL_GOAL_COORD,
                constants.MAX_REL_GOAL_COORD,
            )
        )

        state = state.at[:, :, 5].set(
            self.normalize_tensor(
                state[:, :, 5],
                constants.MIN_REL_GOAL_COORD,
                constants.MAX_REL_GOAL_COORD,
            )
        )
        indices = jnp.array([0, 1, 2, 4, 5, 6])
        state = state[:, :, indices]

        return state

    def process_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: jnp.array of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        """

        # Speed
        obs = obs.at[:, :, :, 0].divide(constants.MAX_SPEED)

        # Relative position
        obs = obs.at[:, :, :, 1].set(
            self.normalize_tensor(
                obs[:, :, :, 1],
                constants.MIN_REL_AGENT_POS,
                constants.MAX_REL_AGENT_POS,
            )
        )
        obs = obs.at[:, :, :, 2].set(
            self.normalize_tensor(
                obs[:, :, :, 2],
                constants.MIN_REL_AGENT_POS,
                constants.MAX_REL_AGENT_POS,
            )
        )

        # Orientation (heading)
        obs = obs.at[:, :, :, 3].divide(constants.MAX_ORIENTATION_RAD)

        # Vehicle length and width
        obs = obs.at[:, :, :, 4].divide(constants.MAX_VEH_LEN)
        obs = obs.at[:, :, :, 5].divide(constants.MAX_VEH_WIDTH)

        obs = obs[:, :, :, :6]

        return obs.reshape(self.num_worlds, self.max_agent_count, -1)

    def process_roadgraph(self, obs):
        """Normalize map observation features."""

        # Road point (x, y) coordinates
        obs = obs.at[:, :, :, 0].set(
            self.normalize_tensor(
                obs[:, :, :, 0],
                constants.MIN_RG_COORD,
                constants.MAX_RG_COORD,
            )
        )

        obs = obs.at[:, :, :, 1].set(
            self.normalize_tensor(
                obs[:, :, :, 1],
                constants.MIN_RG_COORD,
                constants.MAX_RG_COORD,
            )
        )

        # Road line segment length
        obs = obs.at[:, :, :, 2].divide(constants.MAX_ROAD_LINE_SEGMENT_LEN)

        # Road scale (width and height)
        obs = obs.at[:, :, :, 3].divide(constants.MAX_ROAD_SCALE)
        obs = obs.at[:, :, :, 4].divide(constants.MAX_ROAD_SCALE)

        # Road point orientation
        obs = obs.at[:, :, :, 5].divide(constants.MAX_ROAD_SCALE)

        # Road types: one-hot encode them
        one_hot_road_type = jax.nn.one_hot(obs[:, :, :, 6], num_classes=7)

        # Concatenate the one-hot encoding with the rest of the features (exclude index 3 and 4)
        obs = jnp.concatenate((obs[:, :, :, :6], one_hot_road_type), axis=-1)

        return obs.reshape(self.num_worlds, self.max_agent_count, -1)


if __name__ == "__main__":

    env_config = EnvConfig(
        dynamics_model="classic", reward_type="weighted_combination"
    )
    render_config = RenderConfig()

    num_worlds = 2
    max_agents = 64

    # Create data loader
    train_loader = SceneDataLoader(
        root="data/processed/examples",
        batch_size=num_worlds,
        dataset_size=100,
        sample_with_replacement=True,
        shuffle=False,
    )

    # Make env
    env = GPUDriveJaxEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=max_agents,  # Number of agents to control
        device="cuda",
    )

    sim_frames = []

    control_mask = env.cont_agent_mask

    next_obs = env.reset()

    for time_step in range(env.episode_len):
        print(f"Time step: {time_step}")

        sim_states = env.vis.plot_simulator_state(
            env_indices=[0],
            zoom_radius=50,
            time_steps=[time_step],
        )
        sim_frames.append(img_from_fig(sim_states[0]))

        rand_action = jax.random.randint(
            key=jax.random.PRNGKey(0),
            shape=(num_worlds, max_agents),
            minval=0,
            maxval=env.action_space.n,
        )

        # Step the environment
        env.step_dynamics(rand_action)

        # Get info
        next_obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

    env.close()

    mediapy.write_video(
        "sim_video.gif", np.array(sim_frames), fps=20, codec="gif"
    )
