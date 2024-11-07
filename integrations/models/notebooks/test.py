import gpudrive
from pygpudrive.env.config import SceneConfig
from pygpudrive.env.scene_selector import select_scenes

if __name__ == "__main__":
    
    dataset = select_scenes(SceneConfig("data/processed/debug/gpudrive", 1))

    sim = gpudrive.SimManager( 
        exec_mode=gpudrive.madrona.ExecMode.CPU, 
        scenes=dataset,
        gpu_id=0,
        params=gpudrive.Parameters(),
    )
    
    
    sim.reset()