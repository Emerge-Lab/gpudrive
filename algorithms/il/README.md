## Imitation learning


### Overview



### Trajectory object

```C
struct Trajectory {
    madrona::math::Vector2 positions[consts::kTrajectoryLength];
    madrona::math::Vector2 velocities[consts::kTrajectoryLength];
    float headings[consts::kTrajectoryLength];
    float valids[consts::kTrajectoryLength];
    Action inverseActions[consts::kTrajectoryLength];
};
```
