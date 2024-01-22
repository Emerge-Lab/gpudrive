import gpudrive
import gymnasium as gym

class GPUDriveEnv(gym.Env):
    def __init__(self, params: gpudrive.Parameters = None):
        self.sim = gpudrive.SimManager(
            exec_mode=gpudrive.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=1,
            auto_reset=True,
            json_path="tests/test.json",
            params= params if params is not None else GPUDriveEnv.makeParams(),
        )
        

    @staticmethod
    def makeRewardParams(rewardType: gpudrive.RewardType = gpudrive.RewardType.DistanceBased, distanceToGoalThreshold: float = 1.0, distanceToExpertThreshold: float = 1.0, polylineReductionThreshold: float = 1.0, observationRadius: float = 100.0):
        reward_params = gpudrive.RewardParams()
        reward_params.rewardType = rewardType
        reward_params.distanceToGoalThreshold = distanceToGoalThreshold
        reward_params.distanceToExpertThreshold = distanceToExpertThreshold 

        params = gpudrive.Parameters()
        params.polylineReductionThreshold = polylineReductionThreshold
        params.observationRadius = observationRadius
        params.rewardParams = reward_params  