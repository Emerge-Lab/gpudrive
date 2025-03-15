# Agent behavior diversity through reward conditioning

## Introduction

This tutorial explains how to use reward conditioning to create diverse agent behaviors in the GPUDrive environment. Inspired by [Robust autonomy emerges from self-play](https://arxiv.org/abs/2502.03349) (Appendix B.3), we condition agents on different reward weights to produce a spectrum of behaviors ranging from cautious to aggressive, all using a single policy network.

## Setting up the environment

First, let's set up the GPUDrive environment:

```python
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.torch_env import GPUDriveTorchEnv

# Create environment config
env_config = EnvConfig(
    # Change this to "reward_conditioned" to enable reward conditioning
    reward_type="reward_conditioned",

    # Add reward bounds for each component
    collision_weight_lb=-1.0,
    collision_weight_ub=-0.1,
    goal_achieved_weight_lb=0.5,
    goal_achieved_weight_ub=2.0,
    off_road_weight_lb=-1.0,
    off_road_weight_ub=-0.1,

    # Default conditioning mode
    condition_mode="random"
)

render_config = RenderConfig()

# Create data loader
train_loader = SceneDataLoader(
    root="data/processed/examples",
    batch_size=2,
    dataset_size=100,
    sample_with_replacement=True,
    shuffle=False,
)

# Create environment
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=64,  # Number of agents to control
    device="cuda",  # Use GPU if available
)

# Get controlled agent mask
control_mask = env.cont_agent_mask

# Reset the environment
obs = env.reset()
```

To enable reward conditioning, set `reward_type="reward_conditioned"` in the environment config.

## Conditioning Modes

We currently support three conditioning modes:

1. **Random mode**: Each agent receives random reward weights within the specified bounds
2. **Preset mode**: Agents use predefined behavior profiles (cautious, aggressive, etc.)
3. **Fixed mode**: Custom reward weights are provided directly

### 1. Random mode (default for training)

During training, you'll typically use random mode to create a diverse set of agent behaviors:

```python
# Random mode is the default, but you can explicitly set it
obs = env.reset(condition_mode="random")

# Alternatively, set it in the config
env_config.condition_mode = "random"
env = GPUDriveTorchEnv(config=env_config, data_loader=train_loader, max_cont_agents=64)
```

In random mode, each agent receives unique reward weights sampled uniformly within the bounds specified in the config. This creates a diverse population of agents with different behaviors, enabling the policy to learn to handle a wide range of scenarios.

### 2. Preset mode (particular agent profiles)

For testing or evaluation, you might want consistent behavior patterns. Preset mode provides predefined agent personalities:

```python
# All agents use the cautious behavior profile
obs = env.reset(condition_mode="preset", agent_type="cautious")

# Available preset types:
# - "cautious": Strong penalties for risk, moderate goal reward
# - "aggressive": Lower penalties, higher goal reward
# - "balanced": Middle ground between cautious and aggressive
# - "risk_taker": Minimal penalties, maximum goal reward
```

This is useful for evaluating policy performance under specific conditions or for demonstrations.

### 3. Fixed mode (for custom behaviors)

For complete control over agent behavior, you can specify exact reward weights:

```python
# Define custom weights [collision, goal, off_road]
custom_weights = torch.tensor([-0.75, 1.5, -0.3])
obs = env.reset(condition_mode="fixed", agent_type=custom_weights)
```

This gives you precise control over agent behavior for specific testing scenarios.


## Incorporating rewards into the observation

For the policy to adapt its behavior based on reward conditioning, the reward weights should be included in the agent's observation. This allows the policy to "know" it's type. By default, if `reward_type == 'reward_conditioned'`, the weights for each of the 3 reward components are added to the `ego_state`.

## Conclusion

Reward conditioning is a simple but powerful technique for creating diverse agent behaviors in simulation. By conditioning agents on different reward weights, we can create a spectrum of behaviors from a single policy, resulting in more varied simulations. We showed how you can get started with reward conditioning in `gpudrive`.
