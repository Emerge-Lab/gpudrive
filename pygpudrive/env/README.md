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


## Important assumptions

- **We use `nan` values for invalid/padding agents.** The number of valid agents differs per scenario. For instance, we can have two maps (scenes), one with 2 agents and the other with 3 agents. Stepping the environment returns tensors of shape `(num_worlds, kMaxAgentCount)`. Scenarios with less than `max_cont_agents` will have `nan` values for padding agents. For instance

```
done = torch.Tensor([0, 0, nan, nan], [0, 0, 0, nan])
```


## References
