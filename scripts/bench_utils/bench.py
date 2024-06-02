from gpudrive import SimManager
from ..sim_utils.creator import SimCreator

from time import perf_counter
import argparse
import csv
import yaml
import torch
import os
from pynvml import *


def get_gpu_memory_usage():
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.free, info.used, info.total

def get_shapes():
    shape = sim.shape_tensor().to_torch()
    useful_num_agents, useful_num_roads = (
        torch.sum(shape[:, 0]).item(),
        torch.sum(shape[:, 1]).item(),
    )  # shape is a tensor of shape (num_envs, 2)
    num_envs = shape.shape[0]

    useful_num_agents = sim.controlled_state_tensor().to_torch().sum().item()

    actual_num_agents = sim.self_observation_tensor().to_torch().shape[1] * num_envs
    actual_num_roads = sim.map_observation_tensor().to_torch().shape[1] * num_envs
    return actual_num_agents, actual_num_roads, useful_num_agents, useful_num_roads, num_envs

def timeit(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func(*args, **kwargs)  # Execute the function but ignore its result
        end = perf_counter()
        return end - start  # Return only the elapsed time
    return wrapper

@timeit
def reset(sim: SimManager, num_envs: int):
    for i in range(num_envs):
        sim.reset(i)
    sim.step()

@timeit
def step(sim: SimManager, num_steps: int, actions: torch.Tensor):
    for i in range(num_steps):
        sim.action_tensor().to_torch().copy_(actions)
        sim.step()

def save_results():
    pass

def run_stress_test(sim: SimManager, config: dict, num_steps: int = 91):
    actual_num_agents, actual_num_roads, useful_num_agents, useful_num_roads, num_envs = get_shapes()
    episode_length = 91
    step_times = []
    reset_times = []
    used, free, available = [], [], []
    for i in range(0, num_steps, episode_length):
        time_to_reset = reset(sim, num_envs)
        time_to_step = step(sim, num_steps)
        reset_times.append(time_to_reset)
        step_times.append(time_to_step)
        f, u, t = get_gpu_memory_usage()
        used.append(u)
        free.append(f)
        available.append(t)

def run_benchmark(
    sim: SimManager, config: dict, profile_memory: bool = False, num_steps: int = 91
):
    actual_num_agents, actual_num_roads, useful_num_agents, useful_num_roads, num_envs = get_shapes()

    time_to_reset = reset(sim, num_envs)
    actions_tensor = torch.randn_like(sim.action_tensor().to_torch())
    time_to_step = step(sim, num_steps,actions_tensor)

    fps = num_steps / time_to_step
    afps = fps * actual_num_agents
    useful_afps = fps * useful_num_agents

    print(
        f"{useful_num_agents=}, {useful_num_roads=}, {num_envs=}, {time_to_reset=}, {num_steps=},{time_to_step=}, {fps=}, {afps=}, {useful_afps=}"
    )
    # check if benchmark_results.csv exists
    file_path = "benchmark_results.csv"
    # Check if the file exists
    if not os.path.exists(file_path):
        # Open the file in write mode and write the header
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "actual_num_agents",
                    "actual_num_roads",
                    "useful_num_agents",
                    "useful_num_roads",
                    "num_envs",
                    "time_to_reset",
                    "time_to_step",
                    "num_steps",
                    "fps",
                    "afps",
                    "useful_afps",
                    "exec_mode",
                    "datasetInitOptions",
                    "experiment"
                ]
            )

    with open("benchmark_results.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                actual_num_agents,
                actual_num_roads,
                useful_num_agents,
                useful_num_roads,
                num_envs,
                time_to_reset,
                time_to_step,
                num_steps,
                fps,
                afps,
                useful_afps,
                config["sim_manager"]["exec_mode"],
                config["parameters"]["datasetInitOptions"],
                config["experiment"]
            ]
        )


if __name__ == "__main__":
    # Export the
    nvmlInit()
    print(f"Driver Version: {nvmlSystemGetDriverVersion()}")
    parser = argparse.ArgumentParser(description="GPUDrive Benchmarking Tool")
    parser.add_argument(
        "--datasetPath",
        type=str,
        help="Path to the config file",
        default="/home/aarav/gpudrive/config.yml",
        required=False,
    )
    parser.add_argument(
        "--numEnvs", type=int, help="Number of environments", default=2, required=False
    )
    parser.add_argument(
        "--profileMemory",
        action="store_true",
        help="Profile memory usage",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--numSteps",
        type=int,
        help="Number of steps to run the simulation for",
        default=91,
        required=False,
    )
    args = parser.parse_args()
    with open(args.datasetPath, "r") as file:
        config = yaml.safe_load(file)
    config["sim_manager"]["num_worlds"] = args.numEnvs
    sim = SimCreator(config)
    if args.profileMemory:
        run_stress_test(sim, config, args.numSteps)
    else:
        run_benchmark(sim, config, args.numSteps)
    # run_benchmark(sim, config, args.profileMemory)
