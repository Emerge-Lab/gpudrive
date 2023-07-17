import madrona_3d_example

from madrona_learn import LearningState

from policy import make_policy, setup_obs

import torch
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--num-channels', type=int, default=1024)

args = arg_parser.parse_args()

sim = madrona_3d_example.SimManager(
    exec_mode = madrona_3d_example.madrona.ExecMode.CUDA if args.gpu_sim else madrona_3d_example.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = False,
)

obs, process_obs_cb, num_obs_features = setup_obs(sim)
policy = make_policy(process_obs_cb, num_obs_features, args.num_channels)

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights)

# Hack to fill out observations: Reset envs and take step to populate envs
# FIXME: just make it possible to populate observations after init
# (IE run subset of task graph after init)
actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

resets = sim.reset_tensor().to_torch()
actions.fill_(5)
resets[:, 0] = 1
sim.step()

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(torch.zeros(
        *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=torch.device('cpu')))

for i in range(args.num_steps):
    with torch.no_grad():
        policy.fwd_actor(actions, cur_rnn_states, cur_rnn_states, *obs)
    print(actions)
    sim.step()
    print(rewards)
