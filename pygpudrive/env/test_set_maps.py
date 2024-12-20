import gpudrive
import random

def sim_init(dataset):
    reward_params = gpudrive.RewardParams()
    reward_params.rewardType = gpudrive.RewardType.DistanceBased 
    reward_params.distanceToGoalThreshold = 1.0 
    reward_params.distanceToExpertThreshold = 1.0  

    params = gpudrive.Parameters()
    params.polylineReductionThreshold = 0.5  
    params.observationRadius = 10.0 
    params.collisionBehaviour = gpudrive.CollisionBehaviour.AgentStop
    params.rewardParams = reward_params 
    params.maxNumControlledAgents = 2 # we are going to use the second vehicle as the controlled vehicle
    params.IgnoreNonVehicles = True
    params.dynamicsModel = gpudrive.DynamicsModel.InvertibleBicycle
    sim = gpudrive.SimManager(
        exec_mode=gpudrive.madrona.ExecMode.CUDA,
        gpu_id=0,
        scenes=dataset,
        params=params
    )

    return sim


if __name__ == '__main__':
    import sys
    import os
    sys.path.append('..')
    
    data_dir = "data/processed/sampled/training"
    
    batch_size = 100 
    
    all_scene_paths = [
        os.path.abspath(os.path.join(data_dir, scene))
        for scene in sorted(os.listdir(data_dir))
        if scene.startswith("tfrecord")
    ]
    
    sim = sim_init(all_scene_paths[:batch_size])
        
    print(f'Total dataset size: {len(all_scene_paths)}')
        
    for i in range(1000):
       
        data_batch = random.sample(
            all_scene_paths, batch_size
        )
    
        print(f'Batch {i}: {len(data_batch)} scenes')
        
        sim.set_maps(data_batch)
        
        print(f'Set new maps')