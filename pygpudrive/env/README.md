# GPU Drive Multi-Agent Environments Documentation

This repository provides base environments for multi-agent reinforcement learning using `torch` and `jax` in to interface with the `gpudrive` simulator. It follows the `gymnasium` environment standards as closely as possible.

## Quick Start

Begin by downloading traffic scenarios from the [Waymo Open Motion Dataset (WOMDB)](https://github.com/waymo-research/waymo-open-dataset) and save them in a directory. Here, we named our directly `waymo_data`.

Configure the environment using the basic settings in `config`:
```Python
config = EnvConfig()
```
This `config` holds essential parameters of the environment.

To initiate the environment with one world and a maximum of three controllable agents per episode, set the `data_dir` and `device`:
```Python
env = Env(
    config=config,
    num_worlds=1,
    max_cont_agents=3,
    data_dir="waymo_data",
    device="cuda",
)
```

Step the environment using:
```Python
obs, reward, done, info = env.step_dynamics(action)
```

Further configuration details are available in `config.py`.

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

## Rewards

A reward of +1 is assigned when an agent is within the `dist_to_goal_threshold` from the goal, marking the end of the expert trajectory for that vehicle.

## Starting State

Upon initialization, every vehicle starts at the beginning of the expert trajectory.

## Rendering

TODO(dc)

## Sharp Bits

TODO(dc)

## Citations

The Waymo Open Dataset is described in the following publication:

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

This documentation provides concise, clear
