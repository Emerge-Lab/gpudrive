# Navigation in multi-agent traffic scenarios

This base environment provides a `gymnasium` environment that interfaces with the `gpudrive` simulator.

[TODO: Insert gif here]

## Description


## Quick start

Download a number of traffic scenarios from the [Waymo Open Motion Dataset (WOMDB)](https://github.com/waymo-research/waymo-open-dataset) and place them in a directory. Let's call our folder `waymo_data`.

We configure our environment with the basic `config`:
```Python
config = EnvConfig()
```
more on that later.

Once you have the data, we can create the environment. Below, we are making an environment with 1 world and choose to control a maximum of 3 agents per episode. The number of worlds sets the parallism, it defines from how many environments we want to learn and act in at the same time. We also set our `'device'`.

```Python
env = Env(
    config=config,
    num_worlds=1,
    max_cont_agents=3,
    data_dir="waymo_data",
    device="cuda",
)
```

Stepping the the environment results in

```Python
obs, reward, done, info = env.step(action)
```

Next, we discuss how to configure the environment, which is all done in `config.py`.

## Action space


## Observation space


## Rewards


## Starting state

The car starts at rest in the position of xthe expert vehicle.

## Arguments


## Notes

- We use "nan" values to indicate invalid agents in environments. Since the number of valid agents varies per scenario (map), scenarios with fewer than `kMaxAgentCount` controlled agents are padded with `nan` values. For example, suppose we have two scenes, one with two agents and the other with three agents. When we step the environment, we get tensors of shape `(num_worlds, kMaxAgentCount)`, where we can control at most `max_cont_agents` per environment. When we step the environment, the done tensor with `kMaxAgentCount` is five and `max_cont_agents` to three may look like:

```
done = torch.Tensor(
    [0, 1, nan, nan, 0],
    [0, 0, nan, 0, nan],
)
```

The above tensor is interpreted as follows:
- We have one valid agent that is done (as marked by the value `1`)
- We have five valid agents that are not done (marked by `0`)
- We have four invalid agents (as marked by the value `nan`)

## References
