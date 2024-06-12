from algorithms.sb3.ppo import ippo
from pygpudrive.env.config import *
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

import argparse
import imageio
import os
from tqdm import tqdm

TOTAL_NUM_STEPS = 91
torch.set_float32_matmul_precision('high')

def load_policy(policy_path: str, device: str):
    policy = ippo.IPPO.load(policy_path)
    policy.policy.to(device)
    # torch.compile(policy.policy)
    return policy

def init_Env(env_config: EnvConfig):
    env = SB3MultiAgentEnv(
        config=env_config,
        num_worlds=args.numWorlds,
        max_cont_agents=env_config.num_controlled_vehicles,
        data_dir=args.datasetPath,
        device=args.device,
    )
    return env

@torch.compile()
def evaluate(warmup: bool = False):
    obs = env.reset()
    # env._env.sim.step() # Throwaway step 
    world_frames = {}
    for i in range(args.numWorlds):
        world_frames[i] = list()
    for step in tqdm(range(TOTAL_NUM_STEPS)):
        actions, _, _ = policy.policy(obs)
        obs, _, _, _ = env.step(actions.float())
        if(args.disableRender or warmup):
            continue
        for i in range(args.numWorlds):
            world_frames[i].append(env._env.render(i))

    if(args.disableRender or warmup):
        return
    os.makedirs(args.renderPath, exist_ok=True)

    for i in range(args.numWorlds):
        render_path = os.path.join(args.renderPath, f"world_{i}.gif")
        imageio.mimsave(render_path, world_frames[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyPath", type=str, required=True)
    parser.add_argument("--datasetPath", type=str, required=True)
    parser.add_argument("--numWorlds", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disableRender", action="store_true", default=False)
    parser.add_argument("--renderPath", type=str, default="renders/")
    
    args = parser.parse_args()

    policy = load_policy(args.policyPath, args.device)

    env_config = EnvConfig(sample_method="random_n")
    render_config = RenderConfig(
        render_mode=RenderMode.PYGAME_ABSOLUTE,
        view_option=PygameOption.RGB,
        resolution=(1920, 1024),
    )
    env_config.render_config = render_config
    env = init_Env(env_config)

    evaluate(warmup=True) # Warmup
    evaluate() 
