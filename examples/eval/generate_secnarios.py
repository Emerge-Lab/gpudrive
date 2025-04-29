import argparse
import torch
import dataclasses
import os
import sys
import mediapy
import copy
from tqdm import tqdm
from typing import Iterator, List

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.utils.checkpoint import load_agent

from tqdm import tqdm

# WOSAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waymo_open_dataset.protos import sim_agents_submission_pb2


@dataclasses.dataclass
class DeterministicSceneDataLoader:
    root: str
    batch_size: int
    dataset_size: int = None
    file_prefix: str = "tfrecord"

    """
    A data loader for sampling batches of traffic scenarios from a directory of files.

    Attributes:
        root (str): Path to the directory containing scene files.
        batch_size (int): Number of scenes per batch (usually equal to number of worlds in the env).
        dataset_size (int): Maximum number of files to include in the dataset.
        sample_with_replacement (bool): Whether to sample files with replacement.
        file_prefix (str): Prefix for scene files to include in the dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
        shuffle (bool): Whether to shuffle the dataset before batching.
    """

    def __post_init__(self):
        # Validate the path
        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"The specified path does not exist: {self.root}"
            )

        # Create the dataset from valid files in the directory
        self.dataset = [
            os.path.join(self.root, scene)
            for scene in sorted(os.listdir(self.root))
            if scene.startswith(self.file_prefix)
        ]


        # Adjust dataset size based on the provided dataset_size
        if self.dataset_size:
            self.dataset = self.dataset[
                : min(self.dataset_size, len(self.dataset))
            ]

        # If dataset_size < batch_size, repeat the dataset until it matches the batch size
        if self.dataset_size < self.batch_size:
            repeat_count = (self.batch_size // self.dataset_size) + 1
            self.dataset *= repeat_count
            self.dataset = self.dataset[: self.batch_size]

        # Initialize state for iteration
        self._reset_indices()

    def _reset_indices(self):
        """Reset indices for sampling."""
        self.indices = list(range(len(self.dataset)))
        self.current_index = 0

    def __iter__(self) -> Iterator[List[str]]:
        self._reset_indices()
        return self

    def __len__(self):
        """Get the number of batches in the dataloader."""
        return len(self.dataset) // self.batch_size + int((len(self.dataset) % self.batch_size) > 0)

    def __next__(self) -> List[str]:
        
        if self.current_index >= len(self.indices):
            raise StopIteration

        end_index = min(
            self.current_index + self.batch_size, len(self.indices)
        )
        batch = self.dataset[self.current_index : end_index] 
        self.current_index = end_index
        
        return batch


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
    valid_batch,
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
    # done_stack = torch.stack(done_list, dim=-1).cpu().numpy()
    id = id.cpu().numpy()
    control_mask = control_mask.cpu().numpy()

    joint_scenes = {}

    for i, scenario_id in enumerate(scenario_ids):
        if not valid_batch[i]:
            continue
        # control_mask_i = id[i] != 0
        control_mask_i = control_mask[i]

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
        if scenario_id not in joint_scenes:
            joint_scenes[scenario_id] = [joint_scene]
        else:
            joint_scenes[scenario_id].append(joint_scene)

    return joint_scenes


def main(args):
    # Create Data Loader
    OUTPUT_ROOT_DIRECTORY = args.output_root_directory
    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)

    val_loader = DeterministicSceneDataLoader(
        root=args.data_json,
        batch_size=args.num_envs,
        dataset_size=args.dataset_size,
        file_prefix="",
    )

    agent = load_agent(
        path_to_cpt=args.agent_path,
    ).to(args.device)

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
        init_steps=args.init_steps,
        goal_behavior="stop",
        add_reference_path=config.add_reference_path,
        add_reference_speed=config.add_reference_speed,
    )

    train_loader = SceneDataLoader(
        root="data/processed/examples",
        batch_size=args.num_envs,
        dataset_size=100,
        sample_with_replacement=False,
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        # data_loader=train_loader,
        data_loader=copy.deepcopy(val_loader), 
        max_cont_agents=env_config.max_controlled_agents, #args.max_agents,
        device=args.device,
    )

    
    num_iter_pre_shard = max(len(val_loader) // args.num_shards, 1)
    print(f"Total {len(val_loader)} iterations and {args.num_shards} shards. Number of iterations per shard: {num_iter_pre_shard}")

    scenario_rollout_shard = []
    output_filenames = []
    shard_index = 0
    

    def save_shard(shard_index, scenario_rollout_shard):
        # Make sure it is .*\.binproto(-\d{5}-of-\d{5}) format
        output_filename = 'submission.binproto-{:05d}-of-{:05d}'.format(shard_index, args.num_shards)
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=scenario_rollout_shard,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name='your_account@test.com',
            unique_method_name='gpudrive',
            authors=['test'],
            affiliation='waymo',
            description='Submission from the Sim Agents tutorial',
            method_link='https://waymo.com/open/',
            # New REQUIRED fields.
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters='24',
            acknowledge_complies_with_closed_loop_requirement=True,
        )
        with open(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename), 'wb') as f:
            f.write(shard_submission.SerializeToString())
        print(f"Saved {len(scenario_rollout_shard)} scenario into shard to {output_filename}")
        return output_filename
    
    for i, data_batch in tqdm(enumerate(val_loader)):
        valid_batch = [True] * len(data_batch)
        if len(data_batch) < args.num_envs:
            num_pad = args.num_envs - len(data_batch)
            # Pad the batch to match the number of environments
            data_batch += [data_batch[-1]] *num_pad
            valid_batch += [False] * num_pad
        env.swap_data_batch(data_batch)
        cur_scenario_rollouts = {}
        for _ in range(args.num_rollouts):
            # try:
            joint_scenes = rollout(
                env=env,
                sim_agent=agent,
                init_steps=args.init_steps,
                num_envs=args.num_envs,
                max_agents=args.max_agents,
                device=args.device,
                valid_batch=valid_batch,
            )
            
            # ToDO: validate joint scenes
            # except Exception as e:
            #     if args.raise_error:
            #         raise e
            #     else:
            #         print(f"Error during rollout: {e}")
            #         continue
            
            for scenario_id, joint_scene in joint_scenes.items():
                if scenario_id not in cur_scenario_rollouts:
                    cur_scenario_rollouts[scenario_id] = joint_scene
                else:
                    cur_scenario_rollouts[scenario_id].extend(joint_scene)

        for scenario_id, scenario_rollout in cur_scenario_rollouts.items():
            scenario_rollout_shard.append(
                sim_agents_submission_pb2.ScenarioRollouts(
                    scenario_id=scenario_id,
                    joint_scenes=scenario_rollout,
                )
            )
                
        # Save the scenario rollouts to shard files
        if i > 0 and i % num_iter_pre_shard == 0:
            output_filenames.append(save_shard(shard_index, scenario_rollout_shard))
            scenario_rollout_shard = []
            shard_index += 1

    # Save the last shard if it has any scenario rollouts
    if len(scenario_rollout_shard) > 0:
        output_filenames.append(save_shard(shard_index, scenario_rollout_shard))

    # Once we have created all the shards, we can package them directly into a
    # tar.gz archive, ready for submission.
    if args.compress:
        import tarfile
        print("Compressing submission files into a tar.gz archive...")
        with tarfile.open(
            os.path.join(OUTPUT_ROOT_DIRECTORY, 'submission.tar.gz'), 'w:gz'
        ) as tar:
            for output_filename in output_filenames:
                tar.add(
                    os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename),
                    arcname=output_filename,
                )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_json", type=str, default="data/processed/validation")
    # parser.add_argument("--data_tfrecord", type=str, default="data/processed/wosac/validation_tfrecord_1")
    parser.add_argument("--output_root_directory", type=str, default="submission")
    parser.add_argument("--agent_path", type=str, default="checkpoints/model_waypoint_rs__S_1__04_23_19_37_26_618_003500.pt")
    parser.add_argument("--num_shards", type=int, default=2) #150)
    parser.add_argument("--dataset_size", type=int, default=10) #
    parser.add_argument("--max_agents", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--init_steps", type=int, default=10)
    parser.add_argument("--raise_error", action="store_true", default=False)
    parser.add_argument("--compress", action="store_true", default=False)
    
    args = parser.parse_args()

    main(args)