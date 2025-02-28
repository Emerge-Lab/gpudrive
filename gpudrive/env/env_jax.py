from itertools import product
import numpy as np
from gymnasium.spaces import Box, Discrete
import jax
import jax.numpy as jnp

from gpudrive.env import constants
from gpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from gpudrive.env.base_env import GPUDriveGymEnv


class GPUDriveJaxEnv(GPUDriveGymEnv):
    """Jax Gym Environment that interfaces with the GPU Drive simulator."""

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
        self.episode_len = 90

        # Environment parameter setup
        params = self._setup_environment_parameters()

        # Initialize simulator with parameters
        self.sim = self._initialize_simulator(params, scene_config)

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]

        # Total number of controlled agents across all worlds
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.get_obs().shape[-1],)
        )
        self._setup_action_space(action_type)

        # Rendering setup
        self.visualizer = self._setup_rendering()

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

        # Nan actions will be ignored, but we need to replace them with zeros
        actions = jnp.nan_to_num(actions, nan=0)

        # Map action indices to action values
        action_values = self.action_keys_tensor[actions]

        # Feed the actual action values to gpudrive
        self.sim.action_tensor().to_jax().at[:, :, :].set(action_values)

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
                ego_states = self.normalize_ego_state(ego_states)
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
                partner_observations = self.normalize_and_flatten_partner_obs(
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
                road_map_observations = self.normalize_and_flatten_map_obs(
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

        # EGO STATE
        ego_states = self._get_ego_state()

        # PARTNER OBSERVATION
        partner_observations = self._get_partner_obs()

        # ROAD MAP OBSERVATION
        road_map_observations = self._get_road_map_obs()

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

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state = state.at[:, :, 0].divide(constants.MAX_SPEED)
        state = state.at[:, :, 1].divide(constants.MAX_VEH_LEN)
        state = state.at[:, :, 2].divide(constants.MAX_VEH_WIDTH)

        # Relative goal coordinates
        state = state.at[:, :, 3].set(
            self.normalize_tensor(
                state[:, :, 3],
                constants.MIN_REL_GOAL_COORD,
                constants.MAX_REL_GOAL_COORD,
            )
        )

        state = state.at[:, :, 4].set(
            self.normalize_tensor(
                state[:, :, 4],
                constants.MIN_REL_GOAL_COORD,
                constants.MAX_REL_GOAL_COORD,
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

        return obs.reshape(self.num_worlds, self.max_agent_count, -1)

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""

        # Road point coordinates
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

    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_NUM_OBJECTS = 128
    NUM_WORLDS = 50

    env_config = EnvConfig()
    render_config = RenderConfig()
    scene_config = SceneConfig(path="data", num_scenes=NUM_WORLDS)

    # MAKE ENV
    env = GPUDriveJaxEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
        device="cuda",
        render_config=render_config,
    )

    # RUN
    obs = env.reset()

    for _ in range(TOTAL_STEPS):

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

    # import imageio
    # imageio.mimsave("world1.gif", frames_1)
    env.close()
