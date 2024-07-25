## 4. Controlling road objects

Agents in the simulator can be controlled by any user specified policy. By default, that is, if no actions are specified, all agents will be stepped using the logged human trajectories. This mode is also referred to as log-replay.

### 4.1 Expert control

To illustrate this, here we run an episode without specifying any actions. 

```Python
actions = None
for time_step in range(EPISODE_LENGTH):
  
    # STEP
    env.step_dynamics(actions)

    obs = env.get_obs()
    reward = env.get_rewards()
    done = env.get_dones()

    # RENDER
    frame = env.render(world_render_idx=0)
    frames.append(frame)

```

results in the following behavior:

<figure>
<img src="/home/emerge/gpudrive/videos/multi_actors_demo_expert_controlled.gif" alt="...", width=80%>
</figure>


### 4.2 Control subset of agents with a policy, the rest with expert-control

### 4.3 Control agents with different policies