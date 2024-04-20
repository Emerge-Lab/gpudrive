import yaml
import gpudrive

def SimCreator(config: dict = None) -> gpudrive.SimManager:
    if(config is None):
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)
    # Initialize RewardParams
    reward_params_config = config['reward_params']
    reward_params = gpudrive.RewardParams()
    reward_params.rewardType = getattr(gpudrive.RewardType, reward_params_config['rewardType'])
    reward_params.distanceToGoalThreshold = reward_params_config['distanceToGoalThreshold']
    reward_params.distanceToExpertThreshold = reward_params_config['distanceToExpertThreshold']

    # Initialize Parameters
    params_config = config['parameters']
    params = gpudrive.Parameters()
    params.polylineReductionThreshold = 1.0
    params.observationRadius = params_config['observationRadius']
    params.datasetInitOptions = getattr(gpudrive.DatasetInitOptions, params_config['datasetInitOptions'])
    params.collisionBehaviour = getattr(gpudrive.CollisionBehaviour, params_config['collisionBehaviour'])
    params.maxNumControlledVehicles = params_config['maxNumControlledVehicles']
    params.rewardParams = reward_params

    # Initialize SimManager with parameters from the config
    sim_manager_config = config['sim_manager']
    sim = gpudrive.SimManager(
        exec_mode=getattr(gpudrive.madrona.ExecMode, sim_manager_config['exec_mode']),
        gpu_id=sim_manager_config['gpu_id'],
        num_worlds=sim_manager_config['num_worlds'],
        auto_reset=sim_manager_config['auto_reset'],
        json_path=sim_manager_config['json_path'],
        params=params
    )
    return sim

if __name__ == "__main__":
    with open("config.yml", 'r') as file:
       config = yaml.safe_load(file)
    sim = SimCreator(config)