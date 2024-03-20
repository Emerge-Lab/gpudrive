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
    if args.binned:
        num_bins = len(os.listdir("/home/aarav/nocturne_data/binned_jsons"))
        for bin in tqdm(range(1, num_bins + 1)):
            modifyConfigToBinned(bin)
            command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {total_num_envs} --datasetPath config.yml"
            subprocess.run(command, shell=True, check=True)
    for numEnvs in tqdm(range(100, total_num_envs + 1, 10), desc="Overall progress", unit="env", position=0):
        if args.randomized:
            total_iters = 10
            for _ in range(1, total_iters):
                command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {numEnvs} --datasetPath config.yml"
                try:
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError:
                    pass
        else:
            command = f"MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs {numEnvs} --datasetPath config.yml"
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError:
                pass
            


def modifyConfigToRandomize(randomize: bool):
    config_path = "config.yml"
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
    
def modifyConfigToBinned(bin: int):
    config_path = "config.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['sim_manager']['json_path'] = f"/home/aarav/nocturne_data/binned_jsons/bin_{bin}"
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPUDrive Benchmarking Tool')
    parser.add_argument('--totalNumEnvs', type=int, help='Number of environments', default=300, required=False)
    parser.add_argument('--randomized', help='Randomize the dataset', action='store_true', required=False)
    parser.add_argument('--binned', help='Use binned dataset', action='store_true', required=False)
    args = parser.parse_args()
    modifyConfigToRandomize(args.randomized)
    run_bench(args.totalNumEnvs, args)