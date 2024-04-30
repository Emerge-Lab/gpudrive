import os
import time
import pandas as pd
import torch
from datetime import datetime
from time import perf_counter
from multiprocessing import Process, Queue
import numpy as np

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.base_environment import Env

os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "./gpudrive_cache"

env_config = EnvConfig(
    ego_state=True,
    road_map_obs=False,
    partner_obs=False,
    max_controlled_agents=400,
    dataset_init="first_n",
)

EPISODE_LENGTH = 90
TOTAL_STEPS = 2000


def make_and_run_env(
    num_worlds,
    env_config,
    device,
    total_steps,
    q,
    data_dir="/home/emerge/gpudrive/formatted_json_v2_no_tl_train/",
):
    """Make env and take random actions."""

    env = Env(
        config=env_config,
        num_worlds=num_worlds,
        max_cont_agents=env_config.max_controlled_agents,
        data_dir=data_dir,
        device=device,
        auto_reset=False,
    )
    total_resets = 0
    total_step_time = 0
    total_reset_time = 0
    total_world_steps = 0
    total_valid_agent_steps = 0
    total_padding_agent_steps = 0

    obs = env.reset()

    counter = 0
    for _ in range(total_steps):

        rand_actions = torch.randint(
            0, 9, size=(env.num_sims, env.max_cont_agents)
        )

        # Step
        start_step = time.time()

        obs, reward, done, info = env.step(rand_actions)

        end_step = time.time()

        # Store
        total_step_time += end_step - start_step
        total_valid_agent_steps += (~torch.isnan(reward)).sum().item()
        total_padding_agent_steps += reward.flatten().shape[0]
        total_world_steps += 1 * reward.shape[0]

        counter += 1

        if counter == EPISODE_LENGTH:
            start_reset = time.time()
            obs = env.reset()
            end_reset = time.time()

            total_reset_time += end_reset - start_reset
            total_resets += 1 * reward.shape[0]
            counter = 0

    q.put(
        (
            total_step_time,
            total_reset_time,
            total_resets,
            total_world_steps,
            total_valid_agent_steps,
            total_padding_agent_steps,
        )
    )
    env.close()


def run_simulation(num_worlds, env_config, device, total_steps):
    q = Queue()
    p = Process(
        target=make_and_run_env,
        args=(num_worlds, env_config, device, total_steps, q),
    )
    p.start()
    p.join()  # Wait for the process to finish
    return q.get()


if __name__ == "__main__":

    SIM_TYPES = ["multi_agent"]  # , "single_agent"]
    DEVICES = ["cuda"]  # , "cpu"]
    NUM_WORLD_LIST = [1, 50]

    sims = []
    devices = []
    num_envs = []
    time_per_step = []
    time_per_reset = []
    device = []
    env_sps = []
    tot_asps = []
    val_asps = []

    for device in DEVICES:
        for num_worlds in NUM_WORLD_LIST:

            time_before = perf_counter()

            print(
                f"--- Number of worlds: {num_worlds} | on device: {device} ---"
            )

            result = run_simulation(
                num_worlds, env_config, device=device, total_steps=TOTAL_STEPS
            )
            (
                total_step_time,
                total_reset_time,
                total_resets,
                total_world_steps,
                tot_valid_agent_steps,
                tot_padding_agent_steps,
            ) = result

            time_after = perf_counter()

            # Record speed
            total_time = time_after - time_before

            print(f"{total_world_steps / total_time:,.1f} world steps / s")
            print(
                f"{tot_valid_agent_steps / total_time:,.1f} valid agent steps / s"
            )
            print(
                f"{tot_padding_agent_steps / total_time:,.1f} pad agent steps / s \n"
            )

            # Store
            sims.append(SIM_TYPES[0])  # TODO
            devices.append(device)
            num_envs.append(num_worlds)
            time_per_step.append(
                (total_step_time / total_world_steps) * 1000
            )  # MS
            time_per_reset.append(
                (total_reset_time / total_resets) * 1000
            )  # MS
            env_sps.append(total_world_steps / total_time)
            val_asps.append(tot_valid_agent_steps / total_time)
            tot_asps.append(tot_padding_agent_steps / total_time)

    # Convert to df
    df = pd.DataFrame(
        data={
            "sim_type": sims,
            "device": devices,
            "num_envs (BS)": num_envs,
            "step_time (ms)": time_per_step,
            "reset_time (ms)": time_per_reset,
            "total_time (s)": total_time,
            "env_sps": env_sps,
            "val_agent_sps": val_asps,
            "tot_agent_sps": tot_asps,
        }
    )

    dtime = datetime.now().strftime("%m%d_%H%M")

    df.to_csv(f"speed_benchmark_{dtime}.csv", index=False)
