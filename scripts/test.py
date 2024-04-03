import gpudrive
import torch

# Create an instance of RewardParams
reward_params = gpudrive_.RewardParams()
reward_params.rewardType = gpudrive_.RewardType.DistanceBased  # Or any other value from the enum
reward_params.distanceToGoalThreshold = 1.0  # Set appropriate values
reward_params.distanceToExpertThreshold = 1.0  # Set appropriate values

# Create an instance of Parameters
params = gpudrive_.Parameters()
params.polylineReductionThreshold = 0.5  # Set appropriate value
params.observationRadius = 10.0  # Set appropriate value
params.rewardParams = reward_params  # Set the rewardParams attribute to the instance created above

# Now use the 'params' instance when creating SimManager
sim = gpudrive_.SimManager(
    exec_mode=gpudrive_.madrona.ExecMode.CUDA,
    gpu_id=0,
    num_worlds=1,
    auto_reset=True,
    json_path="waymo_data",
    params=params
)

sim.step()
print(sim.done_tensor().to_torch())
