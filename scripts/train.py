import madrona_3d_example

from madrona_learn import (
        train, profile, TrainConfig, PPOConfig, SimInterface,
        ActorCritic, DiscreteActor, Critic, 
        BackboneShared, BackboneSeparate,
        BackboneEncoder, RecurrentBackboneEncoder,
    )
from madrona_learn.models import (
        MLP, LinearLayerDiscreteActor, LinearLayerCritic,
    )

from madrona_learn.rnn import LSTM, FastLSTM

import torch
import argparse
import math
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--lr', type=float, default=0.01)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--steps-per-update', type=int, default=50)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.3)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--dnn', action='store_true')
arg_parser.add_argument('--num-channels', type=int, default=1024)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--actor-rnn', action='store_true')
arg_parser.add_argument('--critic-rnn', action='store_true')
arg_parser.add_argument('--num-bptt-chunks', type=int, default=1)
arg_parser.add_argument('--profile-report', action='store_true')

args = arg_parser.parse_args()

sim = madrona_3d_example.SimManager(
    exec_mode = madrona_3d_example.madrona.ExecMode.CUDA if args.gpu_sim else madrona_3d_example.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

def setup_obs():
    to_others_tensor = sim.to_other_agents_tensor().to_torch()
    to_buttons_tensor = sim.to_buttons_tensor().to_torch()
    to_goal_tensor = sim.to_goal_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    
    obs_tensors = [
        to_others_tensor,
        to_buttons_tensor,
        to_goal_tensor,
        lidar_tensor,
    ]
    
    num_obs_features = (
        math.prod(to_others_tensor.shape[1:]) +
        math.prod(to_buttons_tensor.shape[1:]) +
        math.prod(to_goal_tensor.shape[1:]) +
        math.prod(lidar_tensor.shape[1:])
    )

    def process_obs(to_others, to_buttons, to_goal, lidar):
        assert(not torch.isnan(to_others).any())
        assert(not torch.isnan(to_buttons).any())
        assert(not torch.isnan(to_goal).any())
        assert(not torch.isnan(lidar).any())
        assert(not torch.isinf(to_others).any())
        assert(not torch.isinf(to_buttons).any())
        assert(not torch.isinf(to_goal).any())
        assert(not torch.isinf(lidar).any())

        return torch.cat([
            to_others.view(to_others.shape[0], -1), 
            to_buttons.view(to_buttons.shape[0], -1), 
            to_goal, 
            lidar,
        ], dim=1)

    return obs_tensors, process_obs, num_obs_features

def update_cb(update_idx, update_time, update_results):
    update_id = update_idx + 1
    if update_id % 10 != 0:
        return

    ppo = update_results.ppo_stats

    print(f"\nUpdate: {update_id}")
    print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
    if args.profile_report:
        profile.report()


move_action_dim = 11
num_channels = args.num_channels
obs, process_obs_cb, num_obs_features = setup_obs()

policy = ActorCritic(
    backbone = BackboneShared(
        process_obs = process_obs_cb,
        encoder = RecurrentBackboneEncoder(
            net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 2,
            ),
            rnn = FastLSTM(
                in_channels = num_channels,
                hidden_channels = num_channels,
                num_layers = 1,
            ),
        ),
    ),
    actor = LinearLayerDiscreteActor(
        [move_action_dim, move_action_dim, move_action_dim], num_channels),
    critic = LinearLayerCritic(num_channels),
)

# Hack to fill out observations: Reset envs and take step to populate envs
# FIXME: just make it possible to populate observations after init
# (IE run subset of task graph after init)
actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

resets = sim.reset_tensor().to_torch()
actions.fill_(move_action_dim // 2)
resets[:, 0] = 1
sim.step()

train(
    dev,
    SimInterface(
            step = lambda: sim.step(),
            obs = obs,
            actions = actions,
            dones = dones,
            rewards = rewards,
    ),
    TrainConfig(
        num_updates = args.num_updates,
        steps_per_update = args.steps_per_update,
        num_bptt_chunks = args.num_bptt_chunks,
        ckpt_dir = args.ckpt_dir,
        lr = args.lr,
        gamma = args.gamma,
        gae_lambda = 0.95,
        ppo = PPOConfig(
            num_mini_batches=1,
            clip_coef=0.2,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_loss_coef,
            max_grad_norm=0.5,
            num_epochs=1,
            clip_value_loss=True,
        ),
        mixed_precision = args.fp16,
    ),
    policy,
    update_cb,
)
