import os
import argparse
import subprocess
from tqdm import tqdm
import yaml

def run_bench(args):
    command_template = "MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python -m scripts.bench_utils.bench --numEnvs {} --datasetPath config.yml"
    
    if args.binned:
        num_bins = len(os.listdir("/home/aarav/nocturne_data/binned_jsons"))
        for bin in range(1, num_bins + 1):
            modifyConfigToBinned(bin)
            subprocess.run(command_template.format(args.totalNumEnvs), shell=True, check=True)
        restoreConfig()
        return
    
    for num_envs in tqdm(range(1, args.totalNumEnvs + 1), desc="Overall progress", unit="env", position=0):
        if args.randomized:
            for _ in range(args.totalNumEnvs + 1 - num_envs):
                subprocess.run(command_template.format(num_envs), shell=True, check=True)
        else:
            subprocess.run(command_template.format(num_envs), shell=True, check=True)

def modifyConfigToRandomize(randomize: bool):
    global CONFIG_PATH
    option = "RandomN" if randomize else "FirstN"
    
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    
    config['parameters']['datasetInitOptions'] = option
    
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(config, file)
    
def modifyConfigToBinned(bin: int):
    global DATASET_PATH, CONFIG_PATH
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    if dataset_path is None:
        dataset_path = config['sim_manager']['json_path']

    config['sim_manager']['json_path'] = f"{dataset_path}/binned_jsons/bin_{bin}"
    
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(config, file)

def restoreConfig():
    global DATASET_PATH, CONFIG_PATH
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['sim_manager']['json_path'] = DATASET_PATH
    
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(config, file)

if __name__ == "__main__":
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description='GPUDrive Benchmarking Tool')
    parser.add_argument('--totalNumEnvs', type=int, help='Num envs to run benchmark upto (default: 150)', default=150, required=False)
    parser.add_argument('--randomized', help='Randomize the dataset. For every number of envs, 100 iterations of benchmark are run using randomized subset of the dataset.', action='store_true', required=False)
    parser.add_argument('--binned', help='Use binned dataset', action='store_true', required=False)
    parser.add_argument('--config_path', type=str, help='Path to the config file', default='config.yml', required=False)
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    assert not (args.randomized and args.binned), "Cannot use both randomized and binned dataset"
    modifyConfigToRandomize(args.randomized)
    run_bench(args)