#!/usr/bin/env python

import os
import dataclasses
import random
import numpy as np
import torch
import mediapy
from pathlib import Path
from box import Box as box
from gpudrive.utils.config import load_config
from gpudrive.env.dataset import SceneDataLoader as scene_data_loader
from gpudrive.env.config import EnvConfig as env_config_cls, RenderConfig as render_config_cls
from gpudrive.env.env_torch import GPUDriveTorchEnv as gpusim_env
from gpudrive.visualize.utils import img_from_fig
from gpudrive.networks.late_fusion import NeuralNet as neural_net

cfg_yaml = 'my_tests/conditioned_agent'
model_dir = 'my_tests'
model_name = 'rew_conditioned_0321'
data_root = 'data/processed/examples'
video_dir = Path('./alpha_videos')
video_dir.mkdir(exist_ok=True)

device = 'cpu'
fps = 15
zoom = 70
max_steps = 600
seed = 123
render_fmt = 'mp4'

alphas = {
    'countersense': torch.tensor([1.0, -1 , 1.0]),
    'weird' : torch.tensor([0.0, 0.0, 0.0]),
    'cautious':   torch.tensor([-2.0, 1.0, -2.0]),
    'balanced':   torch.tensor([-0.75, 1.0, -0.75]),
    'assertive':  torch.tensor([-0.1, 1.5, -0.5]),
    'reckless':   torch.tensor([0.0, 1.0,  0.0]),
}

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

cfg = load_config(cfg_yaml)
cfg.environment.condition_mode = 'fixed'

def load_policy(path_to_cpt, model_name, device, env_config):
    ckpt = torch.load(
        f"{path_to_cpt}/{model_name}.pt",
        map_location=device,
        weights_only=False
    )
    net = neural_net(
        input_dim=ckpt['model_arch']['input_dim'],
        action_dim=ckpt['action_dim'],
        hidden_dim=ckpt['model_arch']['hidden_dim'],
        config=env_config
    ).to(device)
    net.load_state_dict(ckpt['parameters'])
    net.eval()
    return net

policy = load_policy(model_dir, model_name, device, cfg.environment)

loader = scene_data_loader(
    data_root,
    batch_size=1,
    dataset_size=1,
    sample_with_replacement=False,
    shuffle=False,
    seed=seed
)
first_scene = next(iter(loader))
loader.batch = first_scene

env_cfg = dataclasses.replace(
    env_config_cls(),
    norm_obs=cfg.environment.norm_obs,
    dynamics_model=cfg.environment.dynamics_model,
    collision_behavior=cfg.environment.collision_behavior,
    reward_type=cfg.environment.reward_type,
    action_space_steer_disc=cfg.environment.action_space_steer_disc,
    action_space_accel_disc=cfg.environment.action_space_accel_disc,
    condition_mode='fixed'
)
render_cfg = render_config_cls()
render_cfg.render_3d = False

env = gpusim_env(
    config=env_cfg,
    data_loader=loader,
    max_cont_agents=64,
    device=device,
    render_config=render_cfg
)

scene_batch = env.data_batch

@torch.no_grad()
def record(alpha_vec, tag):
    env.swap_data_batch(scene_batch)
    env.reset(condition_mode='fixed', agent_type=alpha_vec.to(device))
    frames = []
    t = 0
    done = env.get_dones()
    while not done.all() and t < max_steps:
        obs = env.get_obs()[env.cont_agent_mask]
        act, *_ = policy(obs, deterministic=True)
        template = torch.zeros(
            (env.num_worlds, env.max_agent_count),
            dtype=torch.int64,
            device=device
        )
        template[env.cont_agent_mask] = act.to(device)
        env.step_dynamics(template)
        fig = env.vis.plot_simulator_state([0], [t], zoom_radius=zoom)[0]
        frames.append(img_from_fig(fig))
        t += 1
        done = env.get_dones()
    out = video_dir / f"{tag}.{render_fmt}"
    codec = 'libx264' if render_fmt == 'mp4' else 'gif'
    mediapy.write_video(out, np.stack(frames), fps=fps, codec=codec)
    print('saved', out)

for name, alpha in alphas.items():
    print(f'→ {name:9}  alpha={alpha.tolist()}')
    record(alpha, name)

env.close()
print('\ndone. check:', video_dir.resolve())
