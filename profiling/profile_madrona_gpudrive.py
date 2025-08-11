import os
import random
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from time import perf_counter
import GPUtil
from tqdm import tqdm

from multiprocessing import Process, Queue

import madrona_gpudrive
import threading

MAX_CONT_AGENTS = 64
EPISODE_LENGTH = 80


def make_sim(
    scenes,
    device,
    actor_type,
    obs_type="classic",
    num_worlds=1,
):
    """Make the gpudrive simulator."""

    # Create an instance of RewardParams
    reward_params = madrona_gpudrive.RewardParams()
    reward_params.rewardType = madrona_gpudrive.RewardType.OnGoalAchieved
    reward_params.distanceToGoalThreshold = 1.0
    reward_params.distanceToExpertThreshold = 1.0

    # Create an instance of Parameters
    params = madrona_gpudrive.Parameters()
    # params.numWorlds = num_worlds
    params.polylineReductionThreshold = 0.5
    params.observationRadius = 10.0
    # Set Road Observation Algorithm
    params.roadObservationAlgorithm = madrona_gpudrive.FindRoadObservationsWith.AllEntitiesWithRadiusFiltering
    # params.roadObservationAlgorithm = madrona_gpudrive.FindRoadObservationsWith.KNearestEntitiesWithRadiusFiltering
    params.collisionBehaviour = madrona_gpudrive.CollisionBehaviour.AgentRemoved
    params.rewardParams = reward_params
    params.IgnoreNonVehicles = False
    params.initOnlyValidAgentsAtFirstStep = False

    if obs_type == "lidar":
        params.enableLidar = True
        params.disableClassicalObs = True
    elif obs_type == "classic":
        params.enableLidar = False
        params.disableClassicalObs = False
    else:
        raise ValueError("Invalid obs_type")

    if actor_type == "random":
        params.maxNumControlledAgents = MAX_CONT_AGENTS
    elif actor_type == "expert-actor":
        params.maxNumControlledAgents = 0
    print("Initialized all parameters")
    sim = madrona_gpudrive.SimManager(
        exec_mode=madrona_gpudrive.madrona.ExecMode.CPU
        if device == "cpu"
        else madrona_gpudrive.madrona.ExecMode.CUDA,
        gpu_id=0,
        scenes=scenes,
        params=params,
    )
    # print("Sim Manager Parameters:")
    # for attr in dir(params):
    #     print(f'ATTRIBUTE: {attr}')
    #     if not attr.startswith("__"):
    #         value = getattr(params, attr)
    #         print(f"{attr:20}: {value}")
    #         if attr == "rewardParams":
    #             print("Reward parameters:")
    #             reward_params = getattr(params, attr)
    #             for attr2 in dir(reward_params):
    #                 if not attr2.startswith("__"):
    #                     value2 = getattr(reward_params, attr2)
    #                     print(f"    {attr2:18}: {value2}")
    return sim

def warmup(sim, batch_size):
    """Warmup the simulator."""

    for episode_count in range(5):
        sim.reset(list(range(batch_size)))
        for _ in range(80):
            sim.step()

def run_speed_bench(
    batch_size,
    max_num_objects,
    actor_type,
    sampled_scenes,
    episode_length,
    do_n_resets,
    device,
    obs_type,
    q,
):
    """
    Profiles gpudrive under different conditions.
    """

    # Storage
    total_step_time = 0
    total_reset_time = 0
    total_valid_frames = 0
    total_agent_frames = 0
    # buffer = []

    # Make simulator
    sim = make_sim(
        scenes=sampled_scenes,
        device=device,
        actor_type=actor_type,
        obs_type=obs_type,
        num_worlds=batch_size,
    )
    print(f"Created simulator with {batch_size} worlds")
    # Warmup
    warmup(sim, batch_size)

    sim.reset(list(range(batch_size)))

    # PROFILE STEPS
    for _ in range(episode_length):
        # Get action tensor
        action_tensor = sim.action_tensor().to_torch()

        # print(f"Action tensor has a shape of (num_worlds, max_num_agents_in_scene, 3): {action_tensor.shape}")

        rand_actions = torch.randint(
            0, 9, size=action_tensor.shape
        )

        # Step
        start_step = time.time()

        # Apply actions
        action_tensor.copy_(rand_actions)

        # Step dynamics
        sim.step()

        # Get info
        obs = sim.self_observation_tensor().to_torch()
        reward = sim.reward_tensor().to_torch()
        done = sim.done_tensor().to_torch()
        info = sim.info_tensor().to_torch()

        end_step = time.time()

        # STORE THROUGHPUT
        total_step_time += end_step - start_step

        # TODO: Store valid object distance
        total_valid_frames += (
            (sim.controlled_state_tensor().to_torch() == 1).sum().item()
        )
        total_agent_frames += (
            sim.controlled_state_tensor().to_torch().flatten().shape[0]
        )
    print(f'Total step time: {total_step_time:.5f} seconds')
    # Store valid objects per scene
    valid_obj_dist = (
        (sim.controlled_state_tensor().to_torch() == 1)
        .sum(axis=1)
        .squeeze()
        .tolist()
    )

    # PROFILE RESETS
    for _ in range(do_n_resets):
        start_reset = time.time()
        sim.reset(list(range(batch_size)))
        end_reset = time.time()
        total_reset_time += end_reset - start_reset

    q.put(
        (
            total_step_time,
            total_reset_time,
            do_n_resets,
            episode_length,
            total_valid_frames,
            total_agent_frames,
            valid_obj_dist,
        )
    )


def run_simulation(
    batch_size,
    max_num_objects,
    actor_type,
    scenes,
    episode_length,
    do_n_resets,
    device,
    obs_type="classic",
):
    q = Queue()
    mem_q = Queue()

    def run_bench_and_monitor_mem(*args):
        max_mem = [0]
        def monitor_mem():
            gpu_id = 0
            while not stop_event.is_set():
                mem = GPUtil.getGPUs()[gpu_id].memoryUsed
                if mem > max_mem[0]:
                    max_mem[0] = mem

        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_mem)
        monitor_thread.start()
        run_speed_bench(*args)
        stop_event.set()
        monitor_thread.join()
        # Check if thread is stopped
        if not monitor_thread.is_alive():
            print("Memory monitor thread has stopped.")
        else:
            print("Memory monitor thread is still running!")
        mem_q.put(max_mem[0])

    p = Process(
        target=run_bench_and_monitor_mem,
        args=(
            batch_size,
            max_num_objects,
            actor_type,
            scenes,
            episode_length,
            do_n_resets,
            device,
            obs_type,
            q,
        ),
    )
    print("Starting process for benchmark...")
    p.start()
    p.join()  # Wait for the process to finish
    result = q.get()
    max_gpu_mem = mem_q.get()
    # Return max GPU memory used along with previous results
    print(f"Max GPU memory used: {max_gpu_mem} MB")
    return (*result, max_gpu_mem)


if __name__ == "__main__":

    DATA_FOLDER = "data/processed/examples"
    # BATCH_SIZE_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    BATCH_SIZE_LIST = [8]
    ACTOR_TYPE = "random" # "expert_actor"
    DEVICE = "cuda"
    DATASET_INIT = "first_n" # or "random"
    OBS_TYPE = "lidar" # or "lidar"

    scenes = [os.path.join(DATA_FOLDER, scene) for scene in os.listdir(DATA_FOLDER)]
    print(f"Scenes: {scenes}")
    # take first 100 scenes
    # scenes = scenes[:128]

    # Get device info
    device_name = GPUtil.getGPUs()[0].name
    device_total_memory = GPUtil.getGPUs()[0].memoryTotal

    # Storage
    tot_step_times = np.zeros(len(BATCH_SIZE_LIST))
    tot_reset_times = np.zeros(len(BATCH_SIZE_LIST))
    tot_resets = np.zeros(len(BATCH_SIZE_LIST))
    tot_steps = np.zeros(len(BATCH_SIZE_LIST))
    tot_valid_frames = np.zeros(len(BATCH_SIZE_LIST))
    tot_agent_frames = np.zeros(len(BATCH_SIZE_LIST))
    valid_obj_dist_lst = []
    max_gpu_mem = np.zeros(len(BATCH_SIZE_LIST))

    pbar = tqdm(BATCH_SIZE_LIST, colour="green")
    for idx, batch_size in enumerate(pbar):
        if DATASET_INIT == "random":
            sampled_scenes = random.sample(scenes, batch_size)
        elif DATASET_INIT == "first_n":
            if batch_size > len(scenes):
                # Repeat scenes as needed to reach batch_size
                repeats = batch_size // len(scenes)
                remainder = batch_size % len(scenes)
                if remainder > 0:
                    sampled_scenes = scenes * repeats + scenes[:remainder]
                else:
                    sampled_scenes = scenes * repeats
            else:
                sampled_scenes = scenes[:batch_size]
        pbar.set_description(
            f"Profiling gpudrive with batch size {batch_size} using {ACTOR_TYPE}"
        )

        result = run_simulation(
            batch_size=batch_size,
            max_num_objects=MAX_CONT_AGENTS,
            actor_type=ACTOR_TYPE,
            scenes=sampled_scenes,
            episode_length=EPISODE_LENGTH,
            do_n_resets=80,
            device=DEVICE,
            obs_type=OBS_TYPE,
        )
        (
            tot_step_times[idx],
            tot_reset_times[idx],
            tot_resets[idx],
            tot_steps[idx],
            tot_valid_frames[idx],
            tot_agent_frames[idx],
            valid_obj_dist,
            max_gpu_mem[idx],
        ) = result

        valid_obj_dist_lst.append(valid_obj_dist)

    # Store results
    dtime = datetime.now().strftime("%d_%H%M")

    df = pd.DataFrame(
        data={
            "simulator": "GPU Drive",
            "device_name": device_name,
            "device_mem": device_total_memory,
            "actors": ACTOR_TYPE,
            "obs_type": OBS_TYPE,
            "batch_size (num envs)": BATCH_SIZE_LIST,
            "avg_time_per_reset (ms)": (tot_reset_times / tot_resets) * 1000,
            "avg_time_per_step (ms)": (tot_step_times / tot_steps) * 1000,
            "max_gpu_mem (MB)": max_gpu_mem,
            "all_agent_fps (throughput)": tot_agent_frames / tot_step_times,
            "val_agent_fps (goodput)": tot_valid_frames / tot_step_times,
            "total_steps": tot_steps,
            "total_resets": tot_resets,
            "tot_step_time (s)": tot_step_times,
            "tot_reset_time (s)": tot_reset_times,
            "val_agent_frames": tot_valid_frames,
            "tot_agent_frames": tot_agent_frames,
        }
    )

    df_metadata = pd.DataFrame(
        data={
            "sim_type": "GPUDrive",
            "device_name": device_name,
            "num_envs (BS)": BATCH_SIZE_LIST,
            "num_valid_objects_per_scene (dist)": valid_obj_dist_lst,
        },
    )

    df.to_csv(f"gpudrive_speed_{dtime}.csv", index=False)
    df_metadata.to_csv(f"gpudrive_metadata_{dtime}.csv", index=False)

    print(
        f"Saved results to gpudrive_speed_{dtime}.csv and gpudrive_metadata_{dtime}.csv"
    )