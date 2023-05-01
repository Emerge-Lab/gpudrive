import gpu_hideseek_python
import madrona_learn
from madrona_learn.model import SharedActorCritic, SmallMLP
import torch
import argparse
import math

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--lr', type=float, default=1e-3)
arg_parser.add_argument('--gamma', type=float, default=0.99)
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
        render_width = 0,
        render_height = 0,
)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
elif torch.backends.mps.is_available() and False:
    dev = torch.device('mps')
else:
    dev = torch.device('cpu')

def setup_obs():
    prep_counter_tensor = sim.prep_counter_tensor().to_torch()
    agent_type_tensor = sim.agent_type_tensor().to_torch()
    agent_data_tensor = sim.agent_data_tensor().to_torch()
    box_data_tensor = sim.box_data_tensor().to_torch()
    ramp_data_tensor = sim.ramp_data_tensor().to_torch()
    
    obs_tensors = [
        prep_counter_tensor,
        agent_type_tensor,
        agent_data_tensor,
        box_data_tensor,
        ramp_data_tensor,
        sim.visible_agents_mask_tensor().to_torch(),
        sim.visible_boxes_mask_tensor().to_torch(),
        sim.visible_ramps_mask_tensor().to_torch(),
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

obs_tensors, process_obs_cb, num_obs_features = setup_obs()

policy = SharedActorCritic(
    process_obs_fn = process_obs_cb,
    core = SmallMLP(num_obs_features, 512),
    actor = SharedActorCritic.DefaultDiscreteActor(512,
        [5, 5, 5, 2, 2]),
    critic = SharedActorCritic.DefaultCritic(512))

madrona_learn.train(madrona_learn.SimConfig(
                        step = lambda: sim.step(),
                        obs = obs_tensors,
                        actions = sim.action_tensor().to_torch(),
                        rewards = sim.reward_tensor().to_torch(),
                        dones = sim.done_tensor().to_torch()),
                    madrona_learn.TrainConfig(
                        num_updates = args.num_updates,
                        gamma = args.gamma,
                        lr = args.lr,
                        steps_per_update = args.steps_per_update),
                    policy,
                    dev = dev)

del sim
