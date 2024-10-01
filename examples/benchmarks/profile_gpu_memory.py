import os
import torch
import matplotlib.pyplot as plt
import nvidia_smi

import gpudrive


def make_sim(
    data_dir,
    num_worlds,
    device,
    max_num_objects,
):
    """Make simulator."""

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
    params.datasetInitOptions = gpudrive.DatasetInitOptions.PadN
    params.rewardParams = reward_params
    params.IgnoreNonVehicles = True
    params.maxNumControlledAgents = max_num_objects

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


def main(
    total_timesteps,
    num_worlds,
    episode_length,
    max_num_objects,
    data_dir,
    device="cuda",
):
    # Storage
    time_checkpoints = []
    free_memory = []
    used_memory = []
    perc_used = []

    # MAKE SIM
    sim = make_sim(
        data_dir=data_dir,
        num_worlds=num_worlds,
        device=device,
        max_num_objects=max_num_objects,
    )

    for sim_idx in range(num_worlds):
        obs = sim.reset(sim_idx)

    pid = os.getpid()
    print(f"PID: {pid}")

    # RUN SIMULATOR
    episode_step = 0
    for global_step in range(total_timesteps):

        rand_actions = torch.randint(
            0, 9, size=(num_worlds, max_num_objects, 3)
        )

        # Apply actions
        sim.action_tensor().to_torch().copy_(rand_actions)

        # Step dynamics
        sim.step()

        episode_step += 1

        # LOG GPU MEMORY
        if global_step % 200 == 0:
            nvidia_smi.nvmlInit()

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            time_checkpoints.append(global_step)
            free_memory.append(memory_info.free)
            used_memory.append(memory_info.used)
            perc_used.append((memory_info.used / memory_info.total) * 100)

            print(
                f"Global step: {global_step} | Perc. memory used: {(memory_info.used / memory_info.total) * 100:.3f} % \n"
            )

        # RESET if episode is done
        if episode_step == episode_length:
            for sim_idx in range(num_worlds):
                obs = sim.reset(sim_idx)
            episode_step = 0

    return time_checkpoints, free_memory, used_memory, perc_used


if __name__ == "__main__":
    (time_checkpoints, free_gpu_mem, used_memory, perc_used,) = main(
        total_timesteps=10_000,
        num_worlds=50,
        episode_length=90,
        max_num_objects=128,
        data_dir="data",
    )

    # Plot stats
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("GPU Memory Profiling")
    axs[0].plot(
        time_checkpoints,
        free_gpu_mem,
        label="Free memory",
        linestyle="-",
        marker=".",
    )
    axs[0].plot(
        time_checkpoints,
        used_memory,
        label="Used memory",
        linestyle="-",
        marker=".",
    )
    axs[1].plot(
        time_checkpoints,
        perc_used,
        label="Perc. GPU memory used",
        linestyle="-",
        marker=".",
        color="red",
    )
    axs[0].set_ylabel("Memory (MB)")
    axs[1].set_ylabel("Percentage %")
    axs[0].set_xlabel("Global steps")
    axs[1].set_xlabel("Global steps")
    axs[0].legend(), axs[1].legend()
    plt.tight_layout()
    plt.savefig("gpu_mem_prof.png", dpi=300)
