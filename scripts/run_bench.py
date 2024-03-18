import os
import argparse
import subprocess
from tqdm import tqdm
import yaml

# def run_bench(total_num_envs: int):
#     for numEnvs in tqdm(range(1, total_num_envs + 1), desc="Overall progress", unit="env"):
#         if(args.randomized):
#             for _ in range(1,100):
#                 command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {numEnvs} --datasetPath config.yml"
#                 subprocess.run(command, shell=True, check=True)
#         else:        
#             command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {numEnvs} --datasetPath config.yml"
#             subprocess.run(command, shell=True, check=True)

def run_bench(total_num_envs: int, args):
    for numEnvs in tqdm(range(1, total_num_envs + 1), desc="Overall progress", unit="env", position=0):
        if args.randomized:
            with tqdm(total=99, desc=f"Inner progress for numEnvs={numEnvs}", unit="run", leave=False, position=1) as pbar:
                for _ in range(1, 100):
                    command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {numEnvs} --datasetPath config.yml"
                    try:
                        subprocess.run(command, shell=True, check=True)
                    except subprocess.CalledProcessError:
                        pass
                    pbar.update(1)
        else:
            command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {numEnvs} --datasetPath config.yml"
            subprocess.run(command, shell=True, check=True)


def modifyConfigToRandomize(randomize: bool):
    if(randomize):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config['parameters']['datasetInitOptions'] = "RandomN"
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
    else:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config['parameters']['datasetInitOptions'] = "FirstN"
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPUDrive Benchmarking Tool')
    parser.add_argument('--totalNumEnvs', type=int, help='Number of environments', default=150, required=False)
    parser.add_argument('--randomized', help='Randomize the dataset', action='store_true', required=False)
    args = parser.parse_args()
    modifyConfigToRandomize(args.randomized)
    run_bench(args.totalNumEnvs, args)