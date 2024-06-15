# GPU Drive Multi-Agent Environments Documentation

This repository provides base environments for multi-agent reinforcement learning using `torch` and `jax` in to interface with the `gpudrive` simulator. It follows the `gymnasium` environment standards as closely as possible.

## Quick Start

Begin by downloading traffic scenarios from the [Waymo Open Motion Dataset (WOMDB)](https://github.com/waymo-research/waymo-open-dataset) and save them in a directory. For convenience, we added a couple of scenes in the `example_data` folder to get started.

The environment configuration settings are found in the `config`:
```Python
config = EnvConfig()
```

For example, this creates an environment with one world and a maximum of three controllable agents per scenario:
```Python
env = GPUDriveJaxEnv(
    config=config,
    num_worlds=1,
    max_cont_agents=3,
    data_dir="example_data",
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
ego_state: bool = True
road_map_obs: bool = True
partner_obs: bool = True
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

How to sample the set of scenarios you want to train on can be set using `sample_method`.

| `sample_method` | Description |
|----------|-------------|
| **first_n** | Takes the first `num_worlds` files. Fails if the number of files is less than `num_worlds`. |
| **random_n** | Randomly selects `num_worlds` files from the dataset. Fails if the number of files is less than `num_worlds`. |
| **pad_n** | Initializes as many files as available up to `num_worlds`, then repeats the first file to pad until `num_worlds` files are loaded. Fails if there are more files than `num_worlds`. |
| **exact_n** | Initializes exactly `num_worlds` files, ensuring that the count matches precisely with no more or less. |


## Rendering

TODO(dc + av)

<a href="https://drive.google.com/uc?export=view&id=1z902uYrzvH2Ud9vlg6GcL5fNoFfAfz1V"><img src="https://drive.google.com/uc?export=view&id=1z902uYrzvH2Ud9vlg6GcL5fNoFfAfz1V" style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture" />

## Sharp Bits

TODO(dc)

## Citations

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
