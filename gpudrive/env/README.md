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
head_tilt_actions: torch.Tensor = torch.Tensor([0])
steer_actions: torch.Tensor = torch.round(torch.linspace(-1.0, 1.0, 13), decimals=3)
accel_actions: torch.Tensor = torch.round(torch.linspace(-3, 3, 7), decimals=3)
```

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