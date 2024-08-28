import gpudrive
import torch

# Create an instance of RewardParams
reward_params = gpudrive.RewardParams()
reward_params.rewardType = gpudrive.RewardType.DistanceBased  # Or any other value from the enum
reward_params.distanceToGoalThreshold = 1.0  # Set appropriate values
reward_params.distanceToExpertThreshold = 1.0  # Set appropriate values

# Create an instance of Parameters
params = gpudrive.Parameters()
params.polylineReductionThreshold = 0.5  # Set appropriate value
params.observationRadius = 10.0  # Set appropriate value
params.collisionBehaviour = gpudrive.CollisionBehaviour.Ignore  # Set appropriate value
params.rewardParams = reward_params  # Set the rewardParams attribute to the instance created above
params.maxNumControlledVehicles = 128
params.roadObservationAlgorithm = gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering

# Now use the 'params' instance when creating SimManager
sim = gpudrive.SimManager(
    exec_mode=gpudrive.madrona.ExecMode.CPU,
    gpu_id=0,
    scenes=["/home/aarav/gpudrive/data/tfrecord-00001-of-01000_307.json", 
            "/home/aarav/gpudrive/data/tfrecord-00003-of-01000_109.json", 
            "/home/aarav/gpudrive/data/tfrecord-00012-of-01000_389.json"],
    params=params
)

road_tensor = sim.agent_roadmap_tensor().to_torch()
nan_indices = torch.nonzero(torch.isnan(road_tensor), as_tuple=True)
for i in range(len(nan_indices[0])):
    print(f"NaN found at index: ({nan_indices[0][i]}, {nan_indices[1][i]}, {nan_indices[2][i]}, {nan_indices[3][i]})")
print(torch.isnan(road_tensor).any())


