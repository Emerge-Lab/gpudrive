from gpudrive import SimManager
from ..sim_utils.creator import SimCreator

from time import perf_counter
import argparse
import csv
import yaml
import torch
import os

def run_benchmark(sim: SimManager, config: dict):
    shape = sim.shape_tensor().to_torch()
    useful_num_agents, useful_num_roads = torch.sum(shape[:, 0]).item(), torch.sum(shape[:, 1]).item() # shape is a tensor of shape (num_envs, 2) 
    num_envs = shape.shape[0]

    actual_num_agents = sim.self_observation_tensor().to_torch().shape[1]
    actual_num_roads = sim.map_observation_tensor().to_torch().shape[1]

    start = perf_counter()
    for i in range(num_envs):
        sim.reset(i)
    time_to_reset = perf_counter() - start

    start = perf_counter()
    for i in range(91):
        sim.step()
    time_to_step = perf_counter() - start

    fps = (91 / time_to_step) * num_envs
    afps = fps * actual_num_agents
    useful_afps = fps * useful_num_agents

    print(f"{useful_num_agents=}, {useful_num_roads=}, {num_envs=}, {time_to_reset=}, {time_to_step=}, {fps=}, {afps=}, {useful_afps=}")
    # check if benchmark_results.csv exists
    file_path = 'benchmark_results.csv'
    # Check if the file exists
    if not os.path.exists(file_path):
        # Open the file in write mode and write the header
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['actual_num_agents', 'actual_num_roads', 'useful_num_agents', 'useful_num_roads', 'num_envs', 'time_to_reset', 'time_to_step', 'fps', 'afps', 'useful_afps', 'exec_mode', 'datasetInitOptions'])
    else:
        with open('benchmark_results.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([actual_num_agents, actual_num_roads, useful_num_agents, useful_num_roads, num_envs, time_to_reset, time_to_step, fps, afps, useful_afps, config['sim_manager']['exec_mode'], config['parameters']['datasetInitOptions']])

if __name__ == "__main__":
    # Export the 
    parser = argparse.ArgumentParser(description='GPUDrive Benchmarking Tool')
    parser.add_argument('--datasetPath', type=str, help='Path to the config file', default='/home/aarav/gpudrive/config.yml', required=False)
    parser.add_argument('--numEnvs', type=int, help='Number of environments', default=2, required=False)
    args = parser.parse_args()
    with open(args.datasetPath, 'r') as file:
        config = yaml.safe_load(file)
    config['sim_manager']['num_worlds'] = args.numEnvs
    sim = SimCreator(config)
    run_benchmark(sim, config)