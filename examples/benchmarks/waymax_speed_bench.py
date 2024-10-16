from time import perf_counter
from typing import Callable, Optional
import dataclasses
import numpy as np
import jax.numpy as jnp
import jax
import GPUtil
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from waymax import dataloader
from waymax import env as _env
from waymax import config as _config
from waymax.datatypes import operations
from waymax import dynamics, datatypes
from waymax.agents import actor_core
from waymax import dynamics
from waymax import agents

MAX_CONT_AGENTS = 128


def create_random_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    speed: Optional[float] = None,
) -> actor_core.WaymaxActorCore:
    """Create actor that takes random actions."""

    def select_action(
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state=None,
        rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        """Computes the actions using the given dynamics model and speed."""
        del params, actor_state  # unused.
        traj_t0 = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )
        if speed is None:
            vel_x = traj_t0.vel_x
            vel_y = traj_t0.vel_y
        else:
            vel_x = speed * jnp.cos(traj_t0.yaw)
            vel_y = speed * jnp.sin(traj_t0.yaw)

        is_controlled = is_controlled_func(state)
        traj_t1 = traj_t0.replace(
            x=traj_t0.x + vel_x * datatypes.TIME_INTERVAL,
            y=traj_t0.y + vel_y * datatypes.TIME_INTERVAL,
            vel_x=vel_x,
            vel_y=vel_y,
            valid=is_controlled[..., jnp.newaxis] & traj_t0.valid,
            timestamp_micros=(
                traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL
            ),
        )

        traj_combined = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=-1), traj_t0, traj_t1
        )
        actions = dynamics_model.inverse(
            traj_combined, state.object_metadata, timestep=0
        )

        rand_data = jax.random.uniform(rng, actions.data.shape)

        actions = actions.replace(data=rand_data)

        # Note here actions' valid could be different from is_controlled, it happens
        # when that object does not have valid trajectory from the previous
        # timestep.
        return actor_core.WaymaxActorOutput(
            actor_state=None,
            action=actions,
            is_controlled=is_controlled,
        )

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name=f"random_actor",
    )

def warmup(env, scenario, f_scan_step, episode_length=80):
    for _ in range(5):
        # Reset the environment with the current scenario
        state = env.reset(scenario)

        # Step through the episode using jax.lax.scan
        _, state_traj = jax.lax.scan(
            f=f_scan_step,
            init=state,
            xs=None,
            length=episode_length,
        )


def run_speed_bench(
    batch_size,
    max_num_objects,
    actor_type,
    episode_length=80,
    do_n_resets=80,
):
    """Profiles waymax under different conditions.

    Args:
        batch_size (int): The number of environments to run in parallel.
        max_num_objects (int): Maximum number of objects in the scene.
        take_n_steps (int): Total number of steps to run in the environment.
        do_n_resets (int): Total number of resets.
        actor_type (int): How to control the objects in the scene. Options are:
            - "random": Randomly control all objects.
            - "expert_actor": Use the expert actor.
            - "constant_speed_actor": Use the constant speed actor.

    Raises:
        ValueError: If the actor type is not recognized.

    Returns:
        Tuple[float, float, int, int, int, int]: The total step time, total reset time,
        total resets, total steps, total valid frames, total agent frames, and valid_obj_dist.
    """
    # Storage
    total_step_time = 0
    total_reset_time = 0
    total_valid_frames = 0
    total_agent_frames = 0
    valid_obj_dist = []
    # buffer = [] # Create a buffer because normally we'd store the trajectories

    def f_scan_resets(scenario, _):
        """Reset environment."""
        state = env.reset(scenario)
        return scenario, state

    MAX_CONTROLLED_ACROSS_SCENES = 128

    def f_scan_step(state, _):
        """Get actions and step env."""

        # Get actions
        outputs = [
            jit_select_action({}, state, None, rng)
            for jit_select_action in jit_select_action_list
        ]
        action = agents.merge_actions(outputs)

        # Compute metrics
        step_return = env.reward(state, action)

        # Transition to next state
        next_state = jit_step(state, action)
        return next_state, next_state

    def f_scan_step_with_obs_comp(state, _):
        """Compute observation, get actions and step env."""

        # Compute observations from state
        global_traj = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )

        # Global traffic light information
        current_global_tl = operations.dynamic_slice(
            state.log_traffic_light, jnp.array(state.timestep, int), 1, axis=-1
        )

        # Get obs for every controlled / valid agent in relative coordinates
        for idx in range(MAX_CONTROLLED_ACROSS_SCENES):

            # Get global roadgraph points
            global_rg = datatypes.roadgraph.filter_topk_roadgraph_points(
                state.roadgraph_points,
                state.sim_trajectory.xy[:, idx, state.timestep, :],
                topk=3,
            )

            # Get agent pose: Position, orientation, and rotation matrix
            pose = datatypes.observation.ObjectPose2D.from_center_and_yaw(
                xy=state.sim_trajectory.xy[:, idx, state.timestep, :],
                yaw=state.sim_trajectory.yaw[:, idx, state.timestep],
                valid=state.sim_trajectory.valid[:, idx, state.timestep],
            )

            # Transform to relative coordinates using agent i's pose
            sim_traj = datatypes.observation.transform_trajectory(
                global_traj, pose
            )
            local_rg = datatypes.observation.transform_roadgraph_points(
                global_rg, pose
            )
            local_tl = datatypes.observation.transform_traffic_lights(
                current_global_tl, pose
            )

            # Unpack traffic lights, there are a maximum of 16 traffic lights per scene
            # Not all traffic lights are valid
            valid_tl_ids = local_tl.valid.reshape(-1)

            tl_valid_states = jnp.where(
                valid_tl_ids, local_tl.state.reshape(-1), 0
            )
            tl_x_valid = jnp.where(valid_tl_ids, local_tl.x.reshape(-1), 0)
            tl_y_valid = jnp.where(valid_tl_ids, local_tl.y.reshape(-1), 0)
            tl_z_valid = jnp.where(valid_tl_ids, local_tl.z.reshape(-1), 0)
            tl_lane_ids_valid = jnp.where(
                valid_tl_ids, local_tl.lane_ids.reshape(-1), 0
            )

            # Construct agent observation
            agent_obs = jnp.concatenate(
                (
                    sim_traj.xyz.reshape(-1),
                    sim_traj.yaw.reshape(-1),
                    sim_traj.vel_xy.reshape(-1),
                    sim_traj.vel_yaw.reshape(-1),
                    local_rg.xyz.reshape(-1),
                    local_rg.dir_xyz.reshape(-1),
                    local_rg.types.reshape(-1),
                    tl_valid_states,
                    tl_x_valid,
                    tl_y_valid,
                    tl_z_valid,
                    tl_lane_ids_valid,
                )
            )

        # Get actions
        outputs = [
            jit_select_action({}, state, None, rng)
            for jit_select_action in jit_select_action_list
        ]
        action = agents.merge_actions(outputs)

        # Compute metrics
        step_return = env.reward(state, action)

        # Transition
        next_state = jit_step(state, action)

        return next_state, next_state

    # Configure
    data_config = dataclasses.replace(
        _config.WOD_1_0_0_TRAINING,
        max_num_objects=max_num_objects,
        batch_dims=(batch_size,),
    )

    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.VALID,
    )

    # Initialize dataset
    data_iter = dataloader.simulator_state_generator(config=data_config)
    next(data_iter)  # Skip first state (jit compiling / warmup)

    # Define environment
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.InvertibleBicycleModel(
            normalize_actions=True,
        ),
        config=env_config,
    )

    obj_idx = jnp.arange(max_num_objects)

    if actor_type == "random":
        # Create random actor that controls all objects
        actor = create_random_actor(
            dynamics_model=dynamics.InvertibleBicycleModel(
                normalize_actions=True,
            ),
            is_controlled_func=lambda state: obj_idx > 0,
        )
    elif actor_type == "expert_actor":
        actor = agents.create_expert_actor(
            dynamics_model=dynamics.InvertibleBicycleModel(),
            is_controlled_func=lambda state: obj_idx > 0,
        )
    elif actor_type == "constant_speed_actor":
        actor = agents.create_constant_speed_actor(
            speed=5.0,
            dynamics_model=dynamics.InvertibleBicycleModel(),
            is_controlled_func=lambda state: obj_idx >= 0,
        )
    else:
        raise ValueError(f"Invalid actor type: {actor_type}")

    actors = [actor]

    # Jit the step and select action functions
    jit_step = jax.jit(env.step)
    jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

    rng = jax.random.PRNGKey(0)
    scenario = next(data_iter)

    #Warmup 
    warmup(env, scenario, f_scan_step, episode_length=episode_length)

    # PROFILE RESETS
    start_reset = perf_counter()
    _, state_traj = jax.lax.scan(
        f=f_scan_resets,
        init=scenario,
        xs=None,
        length=do_n_resets,
    )
    end_reset = perf_counter()
    total_reset_time = end_reset - start_reset

    # PROFILE STEPS
    scenario = next(data_iter)
    MAX_CONTROLLED_ACROSS_SCENES = (
        scenario.object_metadata.is_valid.sum(axis=1).max().item()
    )
    state = env.reset(scenario)
    start_step = perf_counter()
    _, state_traj = jax.lax.scan(
        f=f_scan_step_with_obs_comp,
        init=state,
        xs=None,
        length=episode_length,
    )
    end_step = perf_counter()
    total_step_time += end_step - start_step

    # Store trajectories (We go OOM at BS = 16)
    # buffer.append(state_traj)

    # STORE THROUGHPUT AND GOODPUT
    # Take the total of valid objects for the scenario
    valid_obj_dist.append(
        state_traj.object_metadata.is_valid[0, :, :].sum(axis=1).tolist()
    )

    total_valid_frames += (
        state_traj.object_metadata.is_valid.flatten().sum().item()
    )
    total_agent_frames += state_traj.object_metadata.ids.flatten().shape[0]

    return (
        total_step_time,
        total_reset_time,
        do_n_resets,
        episode_length,
        total_valid_frames,
        total_agent_frames,
        valid_obj_dist,
    )


if __name__ == "__main__":

    BATCH_SIZE_LIST = [1, 2, 4, 8, 16]
    ACTOR_TYPE = "expert_actor"

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
            f"Profiling waymax env with batch size {batch_size} using {ACTOR_TYPE}"
        )

        # Run speed benchmark
        (
            tot_step_times[idx],
            tot_reset_times[idx],
            tot_resets[idx],
            tot_steps[idx],
            tot_valid_frames[idx],
            tot_agent_frames[idx],
            valid_obj_dist,
        ) = run_speed_bench(
            batch_size=batch_size,
            max_num_objects=MAX_CONT_AGENTS,
            actor_type=ACTOR_TYPE,
        )

        valid_obj_dist_lst.append(
            [x for sublist in valid_obj_dist for x in sublist]
        )

    dtime = datetime.now().strftime("%d_%H%M")

    # Save
    df = pd.DataFrame(
        data={
            "simulator": "Waymax",
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

    # Store metadata
    df_metadata = pd.DataFrame(
        data={
            "sim_type": "Waymax",
            "device_name": device_name,
            "num_envs (BS)": BATCH_SIZE_LIST,
            "num_valid_objects_per_scene (dist)": valid_obj_dist_lst,
        },
    )

    df.to_csv(f"waymax_speed_{dtime}.csv", index=False)
    df_metadata.to_csv(f"waymax_metadata_{dtime}.csv", index=False)

    print(
        f"Saved results to waymax_speed_{dtime}.csv and waymax_metadata_{dtime}.csv"
    )
