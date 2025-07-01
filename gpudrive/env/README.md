# `gpudrive` gym environments

## Quick Start

To get started, you can use the example data available in the `data/processed/examples` folder. For instructions on downloading more traffic scenarios, refer to [this link](https://github.com/Emerge-Lab/gpudrive/tree/main?tab=readme-ov-file#dataset-------%EF%B8%8F-).

Configure the environment using the default settings in `config`:

```Python
config = EnvConfig()
```

The `config` object holds all environment parameters. A key configuration is the dynamics model, which determines how the successor state (e.g., position, yaw, velocity) is computed for one or more objects given an action (e.g., steering, acceleration).

The following dynamics models are available:

- **Classic**: A kinematic bicycle model using the center of gravity as the reference point, with 2D actions (acceleration, steering curvature). Used in [Nocturne](https://arxiv.org/pdf/2206.09889).
- **InvertibleBicycleModel**: A kinematically realistic model using 2D actions (acceleration, steering curvature) based on [this source](https://github.com/waymo-research/waymax/tree/main/waymax/dynamics).
- **DeltaLocal**: A position-based model using a 3D action (dx, dy, dyaw) to represent displacement relative to current position and orientation. This model doesn't check for infeasible actions, and large displacements can cause unrealistic behavior. Based on [this source](https://github.com/waymo-research/waymax/tree/main/waymax/dynamics).
- **StateDynamics**: A position-based model using a 10D action (x, y, z, yaw, velocities, angular velocities) that directly sets global coordinates. This model doesn't check for infeasible actions, as referenced [here](https://github.com/waymo-research/waymax/tree/main/waymax/dynamics).

Example of creating an environment with one world and a maximum of three controllable agents per scenario:

```Python
env = GPUDriveTorchEnv(
    config=config,
    num_worlds=1,  # Number of parallel environments
    max_cont_agents=3,  # Maximum number of agents to control per scene
    data_dir="data/processed/examples",  # Path to data folder
)
```

To step through the environment:

```Python
env.step_dynamics(actions)

# Extract information
obs = env.get_obs()
reward = env.get_rewards()
done = env.get_dones()
```

For additional configuration details, see `config.py`.

---

> **Note:** You can filter the information from the agents you control using `env.cont_agent_mask`. This boolean mask has the shape `(num_worlds, max_agents_in_scene)` where `kMaxAgentCount` (default 64) is set in `consts.hpp`. It marks `True` for agents under your control and `False` for all others.

---

## Action Space

### Discrete (default: `action_type='discrete'`)

This generates a grid of possible actions, with the action space depending on the `dynamics_model`.

For instance, with `dynamics_model: str = "classic"`, the default action space is:

```Python
# Action space (joint discrete)
action_space_steer_disc: int = 13
action_space_accel_disc: int = 7
action_space_head_tilt_disc: int = 1

# Type-aware action space settings
use_type_aware_actions: bool = True  # Toggles type-aware action mapping: if False, use vehicle ranges for all agents

# Vehicle action ranges
vehicle_accel_range: Tuple[float, float] = (-4.0, 4.0)  # m/s²
vehicle_steer_range: Tuple[float, float] = (-1.57, 1.57)  # radians

# Cyclist action ranges
cyclist_accel_range: Tuple[float, float] = (-2.5, 2.5)    # m/s²
cyclist_steer_range: Tuple[float, float] = (-2.09, 2.09)  # radians (±120°)

# Pedestrian action ranges
pedestrian_accel_range: Tuple[float, float] = (-1.5, 1.5)  # m/s²
pedestrian_steer_range: Tuple[float, float] = (-3.14, 3.14)  # radians (±180°)

# Head tilt action range
head_tilt_action_range: Tuple[float, float] = (-0.7854, 0.7854)  # radians (±45°)
```
The head tilt action is relevant when the view cone is not 360° (see below).


### Continuous

To use a continuous action space, set `action_type='continuous'`.

## Observation Space

Key observation flags include:

```
ego_state: bool = True  # Indicates ego vehicle state
road_map_obs: bool = True  # Provides road graph data
partner_obs: bool = True  # Includes partner vehicle information
norm_obs: bool = True  # Normalizes observations if true
lidar_obs: bool = True # Use LiDAR data
```

| Observation Feature                     | Shape                                                  | Description                                                                                                                | Features                                                                                                                                                                                                                             |
| --------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ego_state** 🚘                        | `(max_num_agents_in_scene, 7)`                          | Basic ego vehicle information.                                                                                               | Speed, length, width, relative goal position (xy), collision state (1 if collided, 0 otherwise), agent ID                                                                                                                            |
| **partner_obs** 🚗 🚴🏻‍♀️ 🚶          | `(max_num_agents_in_scene, max_num_objects - 1, 11)`    | Information about other agents (vehicles, pedestrians, cyclists) within a certain visibility radius.                         | Speed, relative position (xy), relative orientation, length and width, type (0: None, 1: Vehicle, 2: Pedestrian, 3: Cyclist), agent ID                                                                                                  |
| **road_map_obs** 🛣️ 🛑                 | `(max_num_agents_in_scene, top_k_road_points, 13)`      | Information about the road graph and static road objects.                                                                    | Road segment position (xy), segment length, road point scale (xy), orientation, point type (0: None, 1: RoadLine, 2: RoadEdge, 3: RoadLane, 4: CrossWalk, 5: SpeedBump, 6: StopSign)                                                    |
| **lidar_obs**                          | `(max_num_agents_in_scene, 3, num_lidar_samples, 4)`    | LiDAR data                                                                                                                 | LiDAR rays; number of points can be set by adjusting `numLidarSamples` in `src/consts.hpp`. Default is 30 points.                                                                                                                    |

---

### Restriced Observation Space

By default, _every_ road segment and agent within a radius of 50m of the ego is observable and added to partner_obs and road_map_obs, respectively.

The observation can be restriced in various ways:
- **smaller observation radius**: By default the radius is set to 50m. This can be changed using the `obs_radius` option.

- **smaller view cone**: Instead of observing 360° around the vehicle, the view can be restricted to a cone using the `view_cone_half_angle` config parameter. By default the _half_ angle is set to $\pi = 180°$, i.e. a full 360° view. Using the head tilt action, the heading of the view angle can be changed (within the given range) without changing the heading of the vehicle (the direction of driving). Relevant parameters: `view_cone_half_angle`, `action_space_head_tilt_disc`, `head_tilt_action_range`. Specifying the range and discretization produces a linearly spaced interval. Alternatively, `head_tilt_actions` can be used to pass a list of angles directly. `action_space_head_tilt_disc` does not need to be specified in this case.


- **remove agents that are occluded**: Agents that are not technically not visible because they are hidden behind other agents are not added to partner_obs if `remove_occluded_agents = True`. This is determined by casting rays: If one ray hits the vehicle without occlusion, the agent is considered visible. By default, the 8 corners as well as the midpoints of each edge of the bounding box are checked. The number of sampled points on the edge of the bounding box can be set in `consts.hpp`


### Data Structures

For detailed structures, refer to:

- **Agent Observations**: [`gpudrive/datatypes/observation.py`](https://github.com/Emerge-Lab/gpudrive/blob/main/gpudrive/datatypes/observation.py)
- **Roadgraph**: [`gpudrive/datatypes/roadgraph.py`](https://github.com/Emerge-Lab/gpudrive/blob/main/gpudrive/datatypes/roadgraph.py)

These structures are used in `env_torch.py`.

### LiDAR Usage

> **Using LiDAR only**: If you want to use LiDAR data exclusively as the observation, set `disable_classic_obs = True` to improve simulation performance by disabling classical observations. To ensure only LiDAR observations are returned, set all other flags to `False` in `gpudrive/env/config.py`:

```Python
ego_state: bool = False
road_map_obs: bool = False
partner_obs: bool = False  # Do not include partner vehicle info
norm_obs: bool = False  # Disable observation normalization

# NOTE: If disable_classic_obs is True, the other flags will be ignored.
disable_classic_obs: bool = True  # Disable classical observations
lidar_obs: bool = True  # Use LiDAR observations
```

## Rewards

A reward of +1 is given when an agent is within the `dist_to_goal_threshold` from the goal, marking the end of the expert trajectory for that vehicle.

## Starting State

Each vehicle begins at the start of the expert trajectory when the environment is initialized.

## Dataset

For detailed instructions, refer to tutorial `01`.

### Iterating Through the Waymo Open Motion Dataset

The `swap_data_batch()` method in `gpudrive/env/env_torch.py` reinitializes the simulator with new traffic scenarios:

1. **Scene Re-initialization**: This function updates the simulation maps by calling `self.sim.set_maps(dataset)`, replacing the current scenes with those provided in `dataset`, a list of paths to traffic scenarios.
2. **Controlled Agent Mask Re-initialization**: The controlled agents' mask is updated using `self.get_controlled_agents_mask()`, marking which agents are user-controlled. This depends on the selected traffic scenarios.
3. **Agent Count Update**: The function updates `self.max_agent_count` to reflect the number of controlled agents, recalculating `self.num_valid_controlled_agents_across_worlds`, which indicates the active controlled agents across all scenarios.

For an example of how to use this with IPPO, see the `resample_scenario_batch()` method in `gpudrive/env/wrappers/sb3_wrapper.py`.

## Visualization

Refer to the visualizer tutorial for more information.