# GPU Drive Multi-Agent Environments Documentation

This repository provides base environments for multi-agent reinforcement learning using `torch` and `jax` in to interface with the `gpudrive` simulator. It follows the `gymnasium` environment standards as closely as possible.

## Quick Start

Begin by downloading traffic scenarios from the [Waymo Open Motion Dataset (WOMDB)](https://github.com/waymo-research/waymo-open-dataset) and save them in a directory. To get started we use the available data in the `data` folder.

Configure the environment using the basic settings in `config`:
```Python
config = EnvConfig()
```
This `config` all environment parameters.

For example, this creates an environment with one world and a maximum of three controllable agents per scenario:
```Python
env = GPUDriveTorchEnv(
    config=config,
    num_worlds=1,
    max_cont_agents=3,
    data_dir="data",
)
```

Step the environment using:
```Python
obs, reward, done, info = env.step_dynamics(action)
```

Further configuration details are available in `config.py`.

---
> **❗️** You can filter the information from the agents you control using `env.cont_agent_mask`. This boolean mask is of shape `(num_worlds, kMaxAgentCount)`, where `kMaxAgentCount` defaults to 128 and is set in `consts.hpp`. It marks True for agents under your control and False for all others.
---

## Action Space

### Discrete (default; `action_type='discrete'`)
Generates a grid of possible steering and acceleration actions:
```Python
# Action space (joint discrete)
steer_actions: torch.Tensor = torch.round(
    torch.linspace(-1.0, 1.0, 13), decimals=3
)
accel_actions: torch.Tensor = torch.round(
    torch.linspace(-3, 3, 7), decimals=3
)
```

### Continuous
Not supported currently.

## Observation Space

Key observation flags include:
```
ego_state: bool = True  # Indicates ego vehicle state
road_map_obs: bool = True  # Provides road graph data
partner_obs: bool = True  # Includes partner vehicle information
norm_obs: bool = True  # Normalizes observations if true
```

| Observation Feature | Shape                                      | Description | Features                                         |
|---------------------|--------------------------------------------|-------------|--------------------------------------------------|
| **ego_state** 🚘   | `(max_num_objects, 6)`                     |  Basic ego information.           | vehicle speed, vehicle length, vehicle width, relative goal position (xy), collision state (1 if collided, 0 otherwise) |
| **partner_obs**  🚗 🚴🏻‍♀️ 🚶 | `(max_num_objects, max_num_objects - 1, 10)` | Information about the other agents in the environment (vehicles, pedestrians, cyclists) within a certain visibility radius.   | speed of other vehicles, relative position of other vehicles (xy), relative orientation of other vehicles, length and width of other vehicles, type of other vehicle `(0: _None, 1: Vehicle, 2: Pedestrian, 3: Cyclist)` |
| **road_map_obs** 🛣️ 🛑  | `(max_num_objects, top_k_road_points, 13)`  | Information about the road graph  and other static road objects.   | road segment position (xy), road segment length , road point scale (xy), road point orientation, road point type `(0: _None, 1: RoadLine, 2: RoadEdge, 3: RoadLane, 4: CrossWalk, 5: SpeedBump, 6: StopSign)` |

Note that all observations are already transformed to be in a relative coordinate frame.

## Rewards

A reward of +1 is assigned when an agent is within the `dist_to_goal_threshold` from the goal, marking the end of the expert trajectory for that vehicle.

## Starting State

Upon initialization, every vehicle starts at the beginning of the expert trajectory.

## Dataset

The `SceneConfig` dataclass is used to configure how scenes are selected from a dataset. It has four attributes:

- `path`: The path to the dataset.
- `num_scenes`: The number of scenes to select.
- `discipline`: The method for selecting scenes, defaulting to `SelectionDiscipline.PAD_N`. (See options in Table below)
- `k_unique_scenes`: Specifies the number of unique scenes to select, if applicable.

| `discipline` | Description |
|----------|-------------|
| **first_n** | Takes the first `num_worlds` files. Fails if the number of files is less than `num_worlds`. |
| **random_n** | Randomly selects `num_worlds` files from the dataset. Fails if the number of files is less than `num_worlds`. |
| **pad_n** | Initializes as many files as available up to `num_worlds`, then repeats the first file to pad until `num_worlds` files are loaded. Fails if there are more files than `num_worlds`. |
| **exact_n** | Initializes exactly `num_worlds` files, ensuring that the count matches precisely with no more or less. |

## Rendering

Render settings can be changed using the `RenderConfig`.

| `Render Mode` | Description
|--|--|
| **PYGAME_ABSOLUTE** | Renders the absolute view of the scene with all the agents. Returns a single frame for a world.
| **PYGAME_EGOCENTRIC** | Renders the egocentric view for each agent in a scene. Returns `num_agents` frames for each world.
| **PYGAME_LIDAR** | Renders the Lidar views for an egent in a scene if Lidar is enabled. Returns `num_agents` frames for each world.

Resolution of the frames can be specified using the `resolution` param which takes in a tuple of (W,H).

Below are the renders for each mode
<table>
  <tr>
    <td>
      <figure>
        <img src="../../docs/assets/absolute.gif" alt="Absolute">
        <center><figcaption>Absolute</figcaption></center>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../../docs/assets/Egocentric.gif" alt="Egocentric">
        <center><figcaption>Egocentric</figcaption></center>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="../../docs/assets/Lidar360.gif" alt="Lidar with 360 FOV">
        <center><figcaption>Lidar with 360 FOV</figcaption></center>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../../docs/assets/Lidar120.gif" alt="Lidar with 120 FOV">
        <center><figcaption>Lidar with 120 FOV</figcaption></center>
      </figure>
    </td>
  </tr>
</table>

## Sharp Bits

TODO(dc)

## Citations

If you use GPUDrive in your work, please cite us:
TODO(dc)


The Waymo Open Dataset is discussed in the following publication:

```
@misc{ettinger2021large,
      title={Large Scale Interactive Motion Forecasting for Autonomous Driving : The Waymo Open Motion Dataset},
      author={Scott Ettinger and others},
      year={2021},
      eprint={2104.10133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
