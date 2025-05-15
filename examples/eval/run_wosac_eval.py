import torch
import dataclasses
import os
import sys
import mediapy
import logging
import numpy as np
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from box import Box
import yaml

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.metadata import Metadata
from gpudrive.datatypes.info import Info
from gpudrive.utils.checkpoint import load_agent
from gpudrive.visualize.utils import img_from_fig
import madrona_gpudrive

# WOSAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waymo_open_dataset.protos import sim_agents_submission_pb2
from eval.wosac_eval_origin import WOSACMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WOSAC evaluation")
# Suppress excessive logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def get_state(env):
    """Obtain raw agent states."""
    avg_z_pos = Metadata.from_tensor(
        metadata_tensor=env.sim.metadata_tensor(),
        backend=env.backend,
        device=env.device,
    ).avg_z
    ego_state = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )
    mean_xy = env.sim.world_means_tensor().to_torch()[:, :2]
    mean_x = mean_xy[:, 0].unsqueeze(1)
    mean_y = mean_xy[:, 1].unsqueeze(1)
    return (
        ego_state.pos_x + mean_x,
        ego_state.pos_y + mean_y,
        avg_z_pos,  # ego_state.pos_z
        ego_state.rotation_angle,
        ego_state.id,
    )


def rollout(
    env: GPUDriveTorchEnv,
    sim_agent: torch.nn.Module,
    init_steps: int,
    num_envs: int,
    max_agents: int,
    device: str,
    render_simulator_states: bool = False,
    render_agent_pov: bool = False,
    render_every_n_steps: int = 5,
    save_videos: bool = True,
    video_dir: str = "videos",
    video_format: str = "gif",
    guidance_mode: str = "vbd_online",
):
    """Rollout agent in the environment and return the scenario rollouts."""
    # Storage
    env_ids = list(range(num_envs))
    simulator_state_frames = {env_id: [] for env_id in range(num_envs)}
    agent_observation_frames = {env_id: [] for env_id in range(num_envs)}

    start_env_rollout = perf_counter()

    # Zero out actions for parked vehicles
    info = Info.from_tensor(
        env.sim.info_tensor(),
        backend=env.backend,
        device=env.device,
    )

    zero_action_mask = (info.off_road == 1) | (
        info.collided_with_vehicle == 1
    ) & (info.type == int(madrona_gpudrive.EntityType.Vehicle))
    control_mask = env.cont_agent_mask.clone()

    next_obs = env.reset(mask=control_mask)

    # Guidance logging
    num_guidance_points = env.valid_guidance_points
    guidance_densities = num_guidance_points / env.reference_traj_len
    print(
        f"Avg guidance points per agent: {num_guidance_points.cpu().numpy().mean():.2f} which is {guidance_densities.mean().item()*100:.2f} % of the trajectory length (mode = {env.config.guidance_dropout_mode}) \n"
    )

    # Get scenario ids
    scenario_ids_dict = env.get_scenario_ids()
    scenario_ids = list(scenario_ids_dict.values())

    pos_x_list = []
    pos_y_list = []
    pos_z_list = []
    heading_list = []
    done_list = [env.get_dones()]

    # Get initial states
    pos_x, pos_y, pos_z, heading, _ = get_state(env)

    for time_step in range(env.episode_len - init_steps):

        # Predict actions
        action, _, _, _ = sim_agent(next_obs)

        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)

        # Find the integer key for the "do nothing" action (zero steering, zero acceleration)
        # Check using env.action_key_to_values[DO_NOTHING_ACTION_INT]
        DO_NOTHING_ACTION_INT = [
            key
            for key, value in env.action_key_to_values.items()
            if abs(value[0]) == 0.0
            and abs(value[1]) == 0.0
            and abs(value[2]) == 0.0
        ][0]
        action_template[zero_action_mask] = DO_NOTHING_ACTION_INT

        # Step
        env.step_dynamics(action_template)

        # Render
        if render_simulator_states and time_step % render_every_n_steps == 0:
            sim_states = env.vis.plot_simulator_state(
                env_indices=env_ids,
                zoom_radius=100,
                time_steps=[time_step] * len(env_ids),
                plot_guidance_pos_xy=True,
            )
            for idx in range(num_envs):
                simulator_state_frames[idx].append(
                    img_from_fig(sim_states[idx])
                )

        if render_agent_pov and time_step % render_every_n_steps == 0:
            agent_obs = env.vis.plot_agent_observation(
                env_idx=0,
                agent_idx=0,
                figsize=(10, 10),
                trajectory=env.reference_path[0, :, :].to("cpu"),
            )
            agent_observation_frames[idx].append(img_from_fig(agent_obs))

        # Get next observation
        next_obs = env.get_obs(control_mask)
        # NOTE(dc): Make sure to decouple the obs from the reward function
        reward = env.get_rewards()
        done = env.get_dones()

        pos_x, pos_y, pos_z, heading, id = get_state(env)

        pos_x_list.append(pos_x)
        pos_y_list.append(pos_y)
        pos_z_list.append(pos_z)
        heading_list.append(heading)
        done_list.append(done)
    _ = done_list.pop()

    if save_videos:
        for idx in range(num_envs):
            scenario_id = scenario_ids_dict[idx]
            if (
                render_simulator_states
                and len(simulator_state_frames[idx]) > 0
            ):
                mediapy.write_video(
                    f"{video_dir}/{guidance_mode}_sim_state_env_{idx}_{scenario_id}.{video_format}",
                    np.array(simulator_state_frames[idx]),
                    fps=8,
                    codec=video_format,
                )

        if render_agent_pov and len(agent_observation_frames[0]) > 0:
            scenario_id = scenario_ids_dict[0]
            mediapy.write_video(
                f"{video_dir}/{guidance_mode}_agent_0_{scenario_id}.{video_format}",
                np.array(agent_observation_frames[0]),
                fps=8,
                codec=video_format,
            )

    # Generate Scenario
    pos_x_stack = torch.stack(pos_x_list, dim=-1).cpu().numpy()
    pos_y_stack = torch.stack(pos_y_list, dim=-1).cpu().numpy()
    pos_z_stack = torch.stack(pos_z_list, dim=-1).cpu().numpy()
    heading_stack = torch.stack(heading_list, dim=-1).cpu().numpy()
    done_stack = torch.stack(done_list, dim=-1).cpu().numpy()
    id = id.cpu().numpy()
    control_mask = control_mask.cpu().numpy()

    logging.info(
        f"Policy rollout took: {perf_counter() - start_env_rollout:.2f} s (Render = {render_simulator_states}; {len(env.data_batch)} scenarios)."
    )

    start_ground_truth_ext = perf_counter()

    scenario_rollouts = []
    scenario_rollout_masks = []
    for i, scenario_id in enumerate(scenario_ids):
        # control_mask_i = id[i] != 0
        control_mask_i = control_mask[i]
        scenario_rollout_masks.append(done_stack[i, control_mask_i] == 0)
        pos_x_i = pos_x_stack[i, control_mask_i]
        pos_y_i = pos_y_stack[i, control_mask_i]
        pos_z_i = pos_z_stack[i, control_mask_i]
        heading_i = heading_stack[i, control_mask_i]
        id_i = id[i, control_mask_i]

        simulated_trajectories = []
        for a, obj_i_a in enumerate(id_i):
            simulated_trajectories.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=pos_x_i[a],
                    center_y=pos_y_i[a],
                    center_z=pos_z_i[a],
                    heading=heading_i[a],
                    object_id=int(obj_i_a),
                )
            )
        joint_scene = sim_agents_submission_pb2.JointScene(
            simulated_trajectories=simulated_trajectories
        )

        scenario_rollouts.append(
            sim_agents_submission_pb2.ScenarioRollouts(
                joint_scenes=[joint_scene],
                scenario_id=scenario_id,
            )
        )

    logging.info(
        f"Ground truth extraction took: {perf_counter() - start_ground_truth_ext:.2f} s ({len(env.data_batch)} scenarios)."
    )

    return scenario_ids, scenario_rollouts, scenario_rollout_masks


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return config


if __name__ == "__main__":

    # Settings
    MAX_AGENTS = (
        madrona_gpudrive.kMaxAgentCount
    )  # TODO: Set to 128 for real eval
    NUM_ENVS = 100
    DEVICE = "cuda"  # where to run the env rollouts
    NUM_ROLLOUTS_PER_BATCH = 1
    NUM_DATA_BATCHES = 1
    INIT_STEPS = 10
    DATASET_SIZE = 100
    RENDER = False
    LOG_DIR = "examples/eval/figures_data/wosac/"
    GUIDANCE_MODE = (
        "log_replay"  # Options: "vbd_amortized", "vbd_online", "log_replay"
    )
    GUIDANCE_DROPOUT_MODE = "avg"  # Options: "max", "avg", "remove_all"
    GUIDANCE_DROPOUT_PROB = 0.0
    SMOOTHEN_TRAJECTORY = True

    DATA_JSON = "data/processed/wosac/validation/json"
    DATA_TFRECORD = "data/processed/wosac/validation/tfrecord"

    CPT_PATH = "checkpoints/model_guidance_logs__R_10000__05_14_16_54_46_975_000300.pt"

    # Create data loader
    val_loader = SceneDataLoader(
        root=DATA_JSON,
        batch_size=NUM_ENVS,
        dataset_size=DATASET_SIZE,
        sample_with_replacement=True,
        shuffle=True,
        file_prefix="",
        seed=10,
    )

    # Load agent
    agent = load_agent(path_to_cpt=CPT_PATH).to(DEVICE)

    # config = load_config("baselines/ppo/config/ppo_guided_autonomy.yaml")
    # config = config.environment

    config = agent.config

    # Override default environment settings to match those the agent was trained with
    # TODO(dc): Clean this up
    env_config = EnvConfig(
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        guidance_speed_weight=config.guidance_speed_weight,
        guidance_heading_weight=config.guidance_heading_weight,
        smoothness_weight=config.smoothness_weight,
        norm_obs=config.norm_obs,
        add_previous_action=config.add_previous_action,
        guidance=config.guidance,
        add_reference_pos_xy=config.add_reference_pos_xy,
        add_reference_speed=config.add_reference_speed,
        add_reference_heading=config.add_reference_heading,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        goal_behavior=config.goal_behavior,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=False,
        obs_radius=config.obs_radius,
        max_steer_angle=config.max_steer_angle,
        max_accel_value=config.max_accel_value,
        action_space_steer_disc=config.action_space_steer_disc,
        action_space_accel_disc=config.action_space_accel_disc,
        # Override action space
        steer_actions=torch.round(
            torch.linspace(
                -config.max_steer_angle,
                config.max_steer_angle,
                config.action_space_steer_disc,
            ),
            decimals=3,
        ),
        accel_actions=torch.round(
            torch.linspace(
                -config.max_accel_value,
                config.max_accel_value,
                config.action_space_accel_disc,
            ),
            decimals=3,
        ),
        init_mode="wosac_eval",
        init_steps=INIT_STEPS,
        guidance_mode=GUIDANCE_MODE,
        guidance_dropout_prob=GUIDANCE_DROPOUT_PROB,
        guidance_dropout_mode=GUIDANCE_DROPOUT_MODE,
        smoothen_trajectory=SMOOTHEN_TRAJECTORY,
    )

    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=val_loader,
        max_cont_agents=MAX_AGENTS,
        device=DEVICE,
    )

    wosac_metrics = WOSACMetrics(
        save_table_with_baselines=True,
        log_dir=LOG_DIR,
        guidance_mode=GUIDANCE_MODE,
        guidance_density=1.0 - GUIDANCE_DROPOUT_PROB,
    )

    for _ in tqdm(range(NUM_DATA_BATCHES)):
        for _ in range(NUM_ROLLOUTS_PER_BATCH):

            scenario_ids, scenario_rollouts, scenario_rollout_masks = rollout(
                env=env,
                sim_agent=agent,
                init_steps=INIT_STEPS,
                num_envs=NUM_ENVS,
                max_agents=MAX_AGENTS,
                device=DEVICE,
                render_simulator_states=RENDER,
                render_agent_pov=RENDER,
                save_videos=RENDER,
                guidance_mode=GUIDANCE_MODE,
            )
            tf_record_paths = [
                os.path.join(DATA_TFRECORD, f"{scenario_id}.tfrecords")
                for scenario_id in scenario_ids
            ]
            wosac_metrics.update(
                tf_record_paths,
                scenario_rollouts,
                # scenario_rollout_masks=scenario_rollout_masks
            )

        # Swap batch of scenarios
        env.swap_data_batch()

    # Aggregate results
    results = wosac_metrics.compute()

    for key, value in results.items():
        print(f"{key}: {value}")
