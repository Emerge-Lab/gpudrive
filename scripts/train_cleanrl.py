import argparse
import shutil
import yaml
import os
from box import Box

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.traceback import install
from rich.console import Console

import cleanrl
from gpudrive_gym_cleanrl import GPUDriveEnv, make_gpudrive
from policies import LinearMLP

def make_policy(env):
    return LinearMLP(env)


def init_wandb(args, name, id=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': dict(args.train),
            'env': dict(args.env),
            'policy': dict(args.policy),
            #'recurrent': args.recurrent,
        },
        name=name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb

def sweep(args, wandb_name, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(
        sweep=dict(args.sweep),
        project="pufferlib",
    )

    def main():
        try:
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            # TODO: Add update method to namespace
            print(wandb.config.train)
            args.train.__dict__.update(dict(wandb.config.train))
            args.track = True
            train(args, env_module, make_env)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=100)

def train(args):
    args.wandb = None
    if args.track:
        args.wandb = init_wandb(args, args.exp_id, id=args.exp_id)


    env = make_gpudrive(config)
    policy = make_policy(env)

    data = cleanrl.create(config.train, env, policy, wandb=args.wandb)

    while data.global_step < config.train.total_timesteps:
        try:
            cleanrl.evaluate(data)
            cleanrl.train(data)
        except KeyboardInterrupt:
            clean_pufferl.close(data)
            os._exit(0)
        except Exception:
            Console().print_exception()
            os._exit(0)

    clean_pufferl.evaluate(data)
    clean_pufferl.close(data)



if __name__ == '__main__':
    install(show_locals=False) # Rich tracebacks
    # TODO: Add check against old args like --config to demo
    parser = argparse.ArgumentParser(
            description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=True)
    parser.add_argument('--baseline', action='store_true', help='Run baseline')
    parser.add_argument('--mode', type=str, default='train', choices='train eval evaluate sweep autotune baseline profile'.split())
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--no-render', action='store_true', help='Disable render during evaluate')
    parser.add_argument('--exp-id', '--exp-name', type=str, default=None, help="Resume from experiment")
    parser.add_argument('--wandb-entity', type=str, default='jsuarez', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')

    args = parser.parse_args()
    config = Box(yaml.safe_load(open('config.yaml', 'r')))
    print(args.exp_id)
    print(config)
    if args.baseline:
        assert args.mode in ('train', 'eval', 'evaluate')
        args.track = True
        shutil.rmtree(f'experiments/{args.exp_id}', ignore_errors=True)
        run = init_wandb(args, args.exp_id, resume=False)
        if args.mode in ('eval', 'evaluate'):
            model_name = f'{args.exp_id}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(args)
    elif args.mode in ('eval', 'evaluate'):
        try:
            clean_pufferl.rollout(
                make_env,
                args.env,
                agent_creator=make_policy,
                agent_kwargs={'env_module': env_module, 'args': args},
                model_path=args.eval_model_path,
                device=args.train.device
            )
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == 'sweep':
        sweep(args, wandb_name, env_module, make_env)
    elif args.mode == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args.train.env_batch_size)
    elif args.mode == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
