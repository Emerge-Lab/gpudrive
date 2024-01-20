import gpudrive

from gpudrive_learn import LearningState

from policy import make_policy, setup_obs

import torch
import numpy as np
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--action-dump-path', type=str)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')

args = arg_parser.parse_args()

sim = gpudrive.SimManager(
    exec_mode = gpudrive.madrona.ExecMode.CUDA if args.gpu_sim else gpudrive.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
)

obs, num_obs_features = setup_obs(sim)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights)

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

# Flatten N, A, ... tensors to N * A, ...
actions = actions.view(-1, *actions.shape[2:])
dones  = dones.view(-1, *dones.shape[2:])
rewards = rewards.view(-1, *rewards.shape[2:])

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(torch.zeros(
        *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=torch.device('cpu')))

if args.action_dump_path:
    action_log = open(args.action_dump_path, 'wb')
else:
    action_log = None

for i in range(args.num_steps):
    with torch.no_grad():
        action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
        action_dists.best(actions)

        probs = action_dists.probs()

    if action_log:
        actions.numpy().tofile(action_log)

    print()
    print("Self:", obs[0])
    print("Partners:", obs[1])
    print("Room Entities:", obs[2])
    print("Lidar:", obs[3])

    print("Move Amount Probs")
    print(" ", np.array_str(probs[0][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[0][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Move Angle Probs")
    print(" ", np.array_str(probs[1][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[1][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Rotate Probs")
    print(" ", np.array_str(probs[2][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[2][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Grab Probs")
    print(" ", np.array_str(probs[3][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[3][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Actions:\n", actions.cpu().numpy())
    print("Values:\n", values.cpu().numpy())
    sim.step()
    print("Rewards:\n", rewards)

if action_log:
    action_log.close()
