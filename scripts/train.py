import gpu_hideseek_python
import madrona_learn
from madrona_learn.model import (ActorCritic, RecurrentActorCritic,
                                 SmallMLPBackbone, LSTMRecurrentPolicy)
import torch
import argparse
import math

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-hiders', type=int, default=2)
arg_parser.add_argument('--num-seekers', type=int, default=1)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--lr', type=float, default=3e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--steps-per-update', type=int, default=100)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--cpu-sim', action='store_true')

args = arg_parser.parse_args()

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CPU if args.cpu_sim else gpu_hideseek_python.ExecMode.CUDA,
        gpu_id = args.gpu_id,
        num_worlds = args.num_worlds,
        min_entities_per_world = 0,
        max_entities_per_world = 0,
        auto_reset = True,
        render_width = 0,
        render_height = 0,
)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
elif torch.backends.mps.is_available() and False:
    dev = torch.device('mps')
else:
    dev = torch.device('cpu')

def setup_obs(total_agents):
    prep_counter_tensor = sim.prep_counter_tensor().to_torch()[0:total_agents]
    agent_type_tensor = sim.agent_type_tensor().to_torch()[0:total_agents]
    agent_data_tensor = sim.agent_data_tensor().to_torch()[0:total_agents]
    box_data_tensor = sim.box_data_tensor().to_torch()[0:total_agents]
    ramp_data_tensor = sim.ramp_data_tensor().to_torch()[0:total_agents]
    
    obs_tensors = [
        prep_counter_tensor,
        agent_type_tensor,
        agent_data_tensor,
        box_data_tensor,
        ramp_data_tensor,
        sim.visible_agents_mask_tensor().to_torch()[0:total_agents],
        sim.visible_boxes_mask_tensor().to_torch()[0:total_agents],
        sim.visible_ramps_mask_tensor().to_torch()[0:total_agents],
    ]
    
    num_agent_data_features = math.prod(agent_data_tensor.shape[1:])
    num_box_data_features = math.prod(box_data_tensor.shape[1:])
    num_ramp_data_features = math.prod(ramp_data_tensor.shape[1:])

    if dev.type == 'cuda':
        conv_args = {
                'dtype': torch.float16,
                'non_blocking': True,
            }
    else:
        conv_args = {
                'dtype': torch.float32,
            }
    
    def process_obs(prep_counter, agent_type, agent_data, box_data, ramp_data,
                    agent_mask, box_mask, ramp_mask):
        return torch.cat([
                prep_counter.to(**conv_args),
                agent_type.to(**conv_args),
                (agent_data * agent_mask).to(**conv_args).view(
                    -1, num_agent_data_features),
                (box_data * box_mask).to(**conv_args).view(
                    -1, num_box_data_features),
                (ramp_data * ramp_mask).to(**conv_args).view(
                    -1, num_ramp_data_features)
            ], dim=1)

    num_obs_features = prep_counter_tensor.shape[1] + \
        agent_type_tensor.shape[1] + \
        num_agent_data_features + num_box_data_features + \
        num_ramp_data_features

    return obs_tensors, process_obs, num_obs_features

total_agents = args.num_worlds * (args.num_hiders + args.num_seekers)

obs_tensors, process_obs_cb, num_obs_features = setup_obs(total_agents)

policy = RecurrentActorCritic(
    backbone = SmallMLPBackbone(
        process_obs_cb,
        num_obs_features, 512),
    rnn = LSTMRecurrentPolicy(512, 512, 1),
    actor = ActorCritic.DefaultDiscreteActor(512,
        [10, 10, 10, 2, 2]),
    critic = ActorCritic.DefaultCritic(512))


# Hack to fill out observations: Reset envs and take step to populate envs
# FIXME: just make it possible to populate observations after init
# (IE run subset of task graph after init)
resets = sim.reset_tensor().to_torch()
actions = sim.action_tensor().to_torch()[0:total_agents]
dones = sim.done_tensor().to_torch()[0:total_agents]
rewards = sim.reward_tensor().to_torch()[0:total_agents]

actions.zero_()
resets[:, 0] = 1
resets[:, 1] = args.num_hiders
resets[:, 2] = args.num_seekers
sim.step()

madrona_learn.train(madrona_learn.SimData(
                        step = lambda: sim.step(),
                        obs = obs_tensors,
                        actions = actions,
                        dones = dones,
                        rewards = rewards,
                    ),
                    madrona_learn.TrainConfig(
                        num_updates = args.num_updates,
                        gamma = args.gamma,
                        gae_lambda = 0.95,
                        lr = args.lr,
                        steps_per_update = args.steps_per_update,
                        ppo = madrona_learn.PPOConfig(
                            num_mini_batches=1,
                            clip_coef=0.2,
                            value_loss_coef=1.0,
                            entropy_coef=0.01,
                            max_grad_norm=5,
                            num_epochs=1,
                            clip_value_loss=True,
                        ),
                    ),
                    policy,
                    dev = dev)

del sim
