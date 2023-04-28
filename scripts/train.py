import gpu_hideseek_python
import madrona_learn
import torch
import random
import argparse

torch.manual_seed(0)
random.seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-epochs', type=int, required=True)
arg_parser.add_argument('--lr', type=float, default=1e-3)
arg_parser.add_argument('--gamma', type=float, default=0.99)
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CPU,
        gpu_id = args.gpu_id,
        num_worlds = args.num_worlds,
        min_entities_per_world = 0,
        max_entities_per_world = 0,
        render_width = 0,
        render_height = 0,
)

actions = sim.action_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
resets = sim.reset_tensor().to_torch()

madrona_learn.train(lambda: sim.step(),
                    dev=torch.device(f'cuda:{args.gpu_id}'),
                    actions=actions,
                    rewards=rewards,
                    num_epochs=args.num_epochs,
                    gamma=args.gamma,
                    lr=args.lr,
                    steps_per_update=100)

del sim
