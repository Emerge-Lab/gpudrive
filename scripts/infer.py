import madrona_3d_example

from madrona_learn import LearningState

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

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)

arg_parser.add_argument('--num-channels', type=int, default=1024)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')

args = arg_parser.parse_args()

sim = madrona_3d_example.SimManager(
    exec_mode = madrona_3d_example.madrona.ExecMode.CUDA if args.gpu_sim else madrona_3d_example.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
)

obs, process_obs_cb, num_obs_features = setup_obs(sim)
policy = make_policy(process_obs_cb, num_obs_features,
                     args.num_channels, args.separate_value)

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights)

# Hack to fill out observations: Reset envs and take step to populate envs
# FIXME: just make it possible to populate observations after init
# (IE run subset of task graph after init)
actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(torch.zeros(
        *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=torch.device('cpu')))

action_log = open('/tmp/actions', 'wb')

for i in range(args.num_steps):
    with torch.no_grad():
        action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
        action_dists.best(actions)

        probs = action_dists.probs()
        print(probs[0].shape)

    actions.numpy().tofile(action_log)

    print()
    print("Pos:", obs[0])
    print("Others:", obs[1])
    print("Buttons:", obs[2])
    print("Goal:", obs[3])
    print("Lidar:", obs[4])

    print("X Probs")
    print(" ", np.array_str(probs[0][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[0][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Y Probs")
    print(" ", np.array_str(probs[1][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[1][1].cpu().numpy(), precision=2, suppress_small=True))

    print("R Probs")
    print(" ", np.array_str(probs[2][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[2][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Actions:\n", actions.cpu().numpy())
    print("Values:\n", values.cpu().numpy())
    sim.step()
    print("Rewards:\n", rewards)

action_log.close()
