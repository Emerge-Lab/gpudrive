import torch
import dataclasses
import os
import sys
import mediapy
from tqdm import tqdm

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.utils.checkpoint import load_agent

# WOSAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waymo_open_dataset.protos import sim_agents_submission_pb2
# from eval.wosac_eval import WOSACMetrics
from eval.wosac_eval_origin import WOSACMetrics


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
        path_to_cpt="checkpoints/model_waypoint_rs__S_1__04_23_19_37_26_618_003500.pt",
    )

    # Obtain config directly from the agent checkpoint
    config = agent.config

    # Configs
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        norm_obs=config.norm_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        obs_radius=config.obs_radius,
        steer_actions=torch.round(
            torch.linspace(
                -torch.pi / 3, torch.pi / 3, config.action_space_steer_disc
            ),
            decimals=3,
        ),
        accel_actions=torch.round(
            torch.linspace(-4.0, 4.0, config.action_space_accel_disc),
            decimals=3,
        ),
        remove_non_vehicles=config.remove_non_vehicles,
        init_mode="womd_tracks_to_predict",
        init_steps=INIT_STEPS,
        goal_behavior="stop",
        add_reference_path=config.add_reference_path,
        add_reference_speed=config.add_reference_speed,
    )

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
