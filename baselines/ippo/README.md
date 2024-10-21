## Running the IPPO Baseline

To run the multi-agent IPPO baseline using stable-baselines 3 (SB3):

```bash
python baselines/ippo/run_sb3_ppo.py
```

### Resampling the data

The configuration for resampling traffic scenarios includes:

- **`resample_scenarios`**: A boolean that enables or disables traffic scenario resampling when set to `True`.

- **`resample_criterion`**: Set to `"global_step"`, indicating resampling occurs based on the global step count.

- **`resample_freq`**: Specifies resampling frequency at `50,000` steps, recommended to align with `num_worlds * n_steps`.

- **`resample_mode`**: Set to `"random"` for random selection of new scenarios.

```
# RESAMPLE TRAFFIC SCENARIOS
resample_scenarios: bool = True
resample_criterion: str = "global_step" # Options: "global_step"
resample_freq: int = 100_000 # Resample every k steps (recommended to be a multiple of num_worlds * n_steps)
resample_mode: str = "random" # Options: "random"
```

## Implemented networks

### Classic Observations

For classic observations (e.g., `ego_state`), there is support for a permutation equivariant network (recommended). In `baselines/ippo/config.py`, set the following:

```python
# NETWORK
mlp_class = LateFusionNet
policy = LateFusionPolicy
```

The default settings for classic observations are:
```python
ego_state: bool = True  # Use ego vehicle state
road_map_obs: bool = True  # Use road graph data
partner_obs: bool = True  # Include partner vehicle information
norm_obs: bool = True  # Normalize observations
```

### LiDAR Observations

For only LiDAR-based observations, set the following options:

```python
ego_state: bool = False  # Use ego vehicle state
road_map_obs: bool = False  # Use road graph data
partner_obs: bool = False  # Include partner vehicle information
norm_obs: bool = False  # Normalize observations
disable_classic_obs: bool = True  # Disable classic observations for faster sim
lidar_obs: bool = True  # Use LiDAR in observations
```

You can also **mix** classic and LiDAR observations by setting:
```python
ego_state: bool = True  # Include ego vehicle state in observations
road_map_obs: bool = True  # Include road graph in observations
partner_obs: bool = True  # Include partner vehicle info in observations
norm_obs: bool = True  # Normalize observations
disable_classic_obs: bool = False  # Keep classic observations
lidar_obs: bool = True  # Add LiDAR to observations
```

In both cases, you can use a feedforward network from `networks/basic_ffn.py`:
```python
# NETWORK
mlp_class = FFN
policy = FeedForwardPolicy
```
