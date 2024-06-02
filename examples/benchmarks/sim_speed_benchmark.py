import os
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from time import perf_counter
import GPUtil
from tqdm import tqdm

from multiprocessing import Process, Queue

import gpudrive

MAX_CONT_AGENTS = 128
EPISODE_LENGTH = 80

os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "./gpudrive_cache"


def make_sim(
    data_dir,
    num_worlds,
    device,
    actor_type,
):
    """Make the gpudrive simulator."""

    # Create an instance of RewardParams
    reward_params = gpudrive.RewardParams()
    reward_params.rewardType = gpudrive.RewardType.OnGoalAchieved
    reward_params.distanceToGoalThreshold = 1.0
    reward_params.distanceToExpertThreshold = 1.0

    # Create an instance of Parameters
    params = gpudrive.Parameters()
    params.polylineReductionThreshold = 0.5
    params.observationRadius = 10.0
    params.collisionBehaviour = gpudrive.CollisionBehaviour.AgentRemoved
    params.datasetInitOptions = gpudrive.DatasetInitOptions.FirstN
    params.rewardParams = reward_params
    params.IgnoreNonVehicles = True

    if actor_type == "random":
        params.maxNumControlledVehicles = MAX_CONT_AGENTS
    elif actor_type == "expert-actor":
        params.maxNumControlledVehicles = 0

    sim = gpudrive.SimManager(
        exec_mode=gpudrive.madrona.ExecMode.CPU
        if device == "cpu"
        else gpudrive.madrona.ExecMode.CUDA,
        gpu_id=0,
        num_worlds=num_worlds,
        json_path=data_dir,
        params=params,
    )

    return sim


def run_speed_bench(
    batch_size,
    max_num_objects,
    actor_type,
    data_dir,
    episode_length,
    do_n_resets,
    device,
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
        data_dir=data_dir,
        num_worlds=batch_size,
        device=device,
        actor_type=actor_type,
    )

    for sim_idx in range(batch_size):
        obs = sim.reset(sim_idx)
    sim.step()

    # PROFILE STEPS
    for _ in range(episode_length):

        rand_actions = torch.randint(
            0, 9, size=(batch_size, max_num_objects, 3)
        )

        # Step
        start_step = time.time()

        # Apply actions
        sim.action_tensor().to_torch().copy_(rand_actions)

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
        for sim_idx in range(batch_size):
            obs = sim.reset(sim_idx)
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
    data_dir,
    episode_length,
    do_n_resets,
    device,
):
    q = Queue()
    p = Process(
        target=run_speed_bench,
        args=(
            batch_size,
            max_num_objects,
            actor_type,
            data_dir,
            episode_length,
            do_n_resets,
            device,
            q,
        ),
    )
    p.start()
    p.join()  # Wait for the process to finish
    return q.get()


if __name__ == "__main__":

    DATA_FOLDER = "/home/emerge/gpudrive/maps_16"
    BATCH_SIZE_LIST = [1, 2, 4, 8, 16]
    ACTOR_TYPE = "expert_actor"  # "random" #"expert_actor"
    DEVICE = "cuda"

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

    pbar = tqdm(BATCH_SIZE_LIST, colour="green")
    for idx, batch_size in enumerate(pbar):

        pbar.set_description(
            f"Profiling gpudrive with batch size {batch_size} using {ACTOR_TYPE}"
        )

        result = run_simulation(
            batch_size=batch_size,
            max_num_objects=MAX_CONT_AGENTS,
            actor_type=ACTOR_TYPE,
            data_dir=DATA_FOLDER,
            episode_length=EPISODE_LENGTH,
            do_n_resets=80,
            device=DEVICE,
        )
        (
            tot_step_times[idx],
            tot_reset_times[idx],
            tot_resets[idx],
            tot_steps[idx],
            tot_valid_frames[idx],
            tot_agent_frames[idx],
            valid_obj_dist,
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
            "batch_size (num envs)": BATCH_SIZE_LIST,
            "avg_time_per_reset (ms)": (tot_reset_times / tot_resets) * 1000,
            "avg_time_per_step (ms)": (tot_step_times / tot_steps) * 1000,
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
            "sim_type": "Waymax",
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
