## Imitation learning


### Overview


<p align="center">
  <img src="data/imitation_learning.png" width="350" title="Getting started">
</p>


### Extra information

- the trajectory object stores all trajectory information, such as
- expert vehicle positions, velocitires, heading, and if they are valid

```C
struct Trajectory {
    madrona::math::Vector2 positions[consts::kTrajectoryLength];
    madrona::math::Vector2 velocities[consts::kTrajectoryLength];
    float headings[consts::kTrajectoryLength];
    float valids[consts::kTrajectoryLength];
    Action inverseActions[consts::kTrajectoryLength];
};
```