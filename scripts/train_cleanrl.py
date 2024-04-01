from pdb import set_trace as T
import argparse
import shutil
import sys
import os

import importlib
import inspect
import yaml

import pufferlib
import pufferlib.utils

import cleanrl
from cleanrl import rollout

from gpudrive_gym import GPUDriveEnv, make_gpudrive
from sim_utils.creator import SimCreator
   
def make_policy(env, env_module, args):
    policy = env_module.Policy(env)
    if args.force_recurrence or env_module.Recurrent is not None:
        policy = env_module.Recurrent(env, policy, **args.recurrent)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.train.device)

def init_wandb(args, name=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"

    import wandb
    return wandb.init(
        id=args.exp_name or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': args.train,
            'env': args.env,
            'policy': args.policy,
            'recurrent': args.recurrent,
        },
        name=name or args.config,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )

def sweep(args, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(sweep=args.sweep, project="pufferlib")

    def main():
        try:
            args.exp_name = init_wandb(args, env_module)
            if hasattr(wandb.config, 'train'):
                # TODO: Add update method to namespace
                print(args.train.__dict__)
                print(wandb.config.train)
                args.train.__dict__.update(dict(wandb.config.train))
            train(args, env_module, make_env)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=20)

def get_init_args(fn):
    if fn is None:
        return {}
    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'env', 'policy'):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args

def load_from_config():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    default_keys = 'train'.split()
    defaults = {key: config.get(key, {}) for key in default_keys}

    return pufferlib.namespace(**defaults['train'])
   

def train(args, env_module, make_env):
    if args.backend == 'clean_pufferl':
        data = cleanrl.create(
            config=args.train,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            env_creator=make_env,
            env_creator_kwargs=None,
            exp_name=args.exp_name,
            track=args.track,
        )

        while not cleanrl.done_training(data):
            cleanrl.evaluate(data)
            cleanrl.train(data)

        print('Done training. Saving data...')
        cleanrl.close(data)
        print('Run complete')
    elif args.backend == 'sb3':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        from sb3_contrib import RecurrentPPO

        envs = make_vec_env(lambda: make_env(**args.env),
            n_envs=args.train.num_envs, seed=args.train.seed, vec_env_cls=DummyVecEnv)

        model = RecurrentPPO("CnnLstmPolicy", envs, verbose=1,
            n_steps=args.train.batch_rows*args.train.bptt_horizon,
            batch_size=args.train.batch_size, n_epochs=args.train.update_epochs,
            gamma=args.train.gamma
        )

        model.learn(total_timesteps=args.train.total_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse environment argument', add_help=False)
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration in config.yaml to use')
    parser.add_argument('--backend', type=str, default='clean_pufferl', choices=['clean_pufferl', 'sb3'], help='Backend to use')
    parser.add_argument('--mode', type=str, default='train', choices='train sweep evaluate'.split())
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--baseline', action='store_true', help='Baseline run')
    parser.add_argument('--no-render', action='store_true', help='Disable render during evaluate')
    parser.add_argument('--exp-name', type=str, default=None, help="Resume from experiment")
    parser.add_argument('--wandb-entity', type=str, default='jsuarez', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--force-recurrence', action='store_true', help='Force model to be recurrent, regardless of defaults')

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__


    make_env = make_gpudrive
    args = pufferlib.namespace(**args)
    args.train = load_from_config()
    if args.mode == 'sweep':
        args.track = True
    elif args.track:
        args.exp_name = init_wandb(args).id
    elif args.baseline:
        args.track = True
        run = init_wandb(args, name=args.exp_name, resume=False)
        if args.mode == 'evaluate':
            model_name = f'puf-{version}-{args.config}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(args, GPUDriveEnv, make_env)
        exit(0)
    elif args.mode == 'sweep':
        sweep(args, env_module, make_env)
        exit(0)
    elif args.mode == 'evaluate':
        rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.train.device
        )