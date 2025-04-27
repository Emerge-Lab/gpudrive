import torch
import dataclasses
import os
import sys
import mediapy
import logging
from time import perf_counter
from tqdm import tqdm

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.utils.checkpoint import load_agent

# WOSAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waymo_open_dataset.protos import sim_agents_submission_pb2
from eval.wosac_eval_origin import WOSACMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WOSAC evaluation")


def get_state(env):
    """Obtain raw agent states."""
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
        ego_state.pos_z,
        ego_state.rotation_angle,
        ego_state.id,
    )


def rollout(
    env,
    sim_agent,
    init_steps,
    num_envs,
    max_agents,
    device,
):
    """Rollout agent in the environment and return the scenario rollouts."""

    start_env_rollout = perf_counter()

    next_obs = env.reset()

    scenario_ids = list(env.get_scenario_ids().values())

    pos_x_list = []
    pos_y_list = []
    pos_z_list = []
    heading_list = []
    done_list = [env.get_dones()]

    control_mask = env.cont_agent_mask

    pos_x, pos_y, pos_z, heading, _ = get_state(env)

    for time_step in range(env.episode_len - init_steps):

        # Predict actions
        action, _, _, _ = sim_agent(next_obs[control_mask])

        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)

        # Step
        env.step_dynamics(action_template)

        next_obs = env.get_obs()
        done = env.get_dones()

        pos_x, pos_y, pos_z, heading, id = get_state(env)

        pos_x_list.append(pos_x)
        pos_y_list.append(pos_y)
        pos_z_list.append(pos_z)
        heading_list.append(heading)
        done_list.append(done)
    _ = done_list.pop()

    # Generate Scenario
    pos_x_stack = torch.stack(pos_x_list, dim=-1).cpu().numpy()
    pos_y_stack = torch.stack(pos_y_list, dim=-1).cpu().numpy()
    pos_z_stack = torch.stack(pos_z_list, dim=-1).cpu().numpy()
    heading_stack = torch.stack(heading_list, dim=-1).cpu().numpy()
    done_stack = torch.stack(done_list, dim=-1).cpu().numpy()
    id = id.cpu().numpy()
    control_mask = control_mask.cpu().numpy()

    logging.info(
        f"Policy rollout took: {perf_counter() - start_env_rollout:.2f} s ({len(env.data_batch)} scenarios)."
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


if __name__ == "__main__":

    # Settings
    MAX_AGENTS = 64
    NUM_ENVS = 1
    DEVICE = "cpu"
    NUM_BATCHES = 1
    NUM_ROLLOUTS = 10
    INIT_STEPS = 10
    DATASET_SIZE = 100

    DATA_JSON = "data/processed/wosac/validation_json_1"
    DATA_TFRECORD = "data/processed/wosac/validation_tfrecord_1"

    # Create data loader
    val_loader = SceneDataLoader(
        root=DATA_JSON,
        batch_size=NUM_ENVS,
        dataset_size=DATASET_SIZE,
        sample_with_replacement=True,
        file_prefix="",
    )

    # Load agent
    agent = load_agent(
        path_to_cpt="checkpoints/model_waypoint_rs__S_1__04_24_19_08_49_096_000850.pt",
        # path_to_cpt="checkpoints/model_guidance_log_replay__S_100__04_25_14_04_49_149_002500.pt",
    ).to(DEVICE)

    # Override default environment settings to match those the agent was trained with
    default_config = EnvConfig()
    config_dict = {
        field.name: getattr(agent.config, field.name)
        for field in dataclasses.fields(EnvConfig)
        if hasattr(agent.config, field.name)
        and getattr(agent.config, field.name)
        != getattr(default_config, field.name)
    }

    # Add the fixed overrides specific to WOSAC evaluation
    fixed_overrides = {
        "steer_actions": torch.round(
            torch.linspace(
                -torch.pi / 3,
                torch.pi / 3,
                agent.config.action_space_steer_disc,
            ),
            decimals=3,
        ),
        "accel_actions": torch.round(
            torch.linspace(-4.0, 4.0, agent.config.action_space_accel_disc),
            decimals=3,
        ),
        "init_mode": "womd_tracks_to_predict",
        "init_steps": INIT_STEPS,
        "goal_behavior": "stop",
    }

    env_config = dataclasses.replace(
        default_config, **config_dict, **fixed_overrides
    )

    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=val_loader,
        max_cont_agents=MAX_AGENTS,
        device=DEVICE,
    )

    wosac_metrics = WOSACMetrics()

    for _ in tqdm(range(NUM_BATCHES)):
        for _ in range(NUM_ROLLOUTS):
            # try:
            scenario_ids, scenario_rollouts, scenario_rollout_masks = rollout(
                env=env,
                sim_agent=agent,
                init_steps=INIT_STEPS,
                num_envs=NUM_ENVS,
                max_agents=MAX_AGENTS,
                device=DEVICE,
            )
            # except Exception as e:
            #     print(f"Error during rollout: {e}")
            #     continue

            tf_record_paths = [
                os.path.join(DATA_TFRECORD, f"{scenario_id}.tfrecords")
                for scenario_id in scenario_ids
            ]
            wosac_metrics.update(
                tf_record_paths,
                scenario_rollouts,
                # scenario_rollout_masks=scenario_rollout_masks
            )
        try:
            env.swap_data_batch()
        except Exception as e:
            break

    results = wosac_metrics.compute()

    for key, value in results.items():
        print(f"{key}: {value}")
