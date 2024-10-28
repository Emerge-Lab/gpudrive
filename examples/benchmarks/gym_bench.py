from time import perf_counter

import torch
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

if __name__ == '__main__':
    
    TOTAL_STEPS = 90
    MAX_CONTROLLED_AGENTS = 128
    NUM_WORLDS = 50

    env_config = EnvConfig(dynamics_model="classic")
    render_config = RenderConfig()
    scene_config = SceneConfig("data/processed/examples", NUM_WORLDS)

    # MAKE ENV
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_CONTROLLED_AGENTS,  # Number of agents to control
        device="cuda",
        render_config=render_config,
    )
    env.reset()

    # Warmup 

    rand_action = torch.Tensor(
        [
            [
                env.action_space.sample()
                for _ in range(
                    env_config.max_num_agents_in_scene * NUM_WORLDS
                )
            ]
        ]
    ).reshape(NUM_WORLDS, env_config.max_num_agents_in_scene)

    for episode in range(10):
        env.reset()
        for step in range(TOTAL_STEPS):
            env.step_dynamics(rand_action)

    env.reset()
    start = perf_counter()
    for _ in range(TOTAL_STEPS):
        env.step_dynamics(rand_action)
        obs = env.get_obs()
    end = perf_counter()

    FPS = TOTAL_STEPS / (end - start)
    AGENT_FPS = FPS * env.num_valid_controlled_agents_across_worlds


    print(f"FPS: {FPS:.2f}")
    print(f"AGENT_FPS: {AGENT_FPS:.2f}")


    env.close()
