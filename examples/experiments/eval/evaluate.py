import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from box import Box
import dataclasses
from datetime import datetime
from pathlib import Path

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.dataset import SceneDataLoader

from networks.late_fusion import LateFusionTransformer


def load_policy(path_to_cpt, device):
    """Load a policy from a given path."""

    # Load the saved checkpoint
    saved_cpt = torch.load(path_to_cpt, map_location=device)

    # Create policy architecture from saved checkpoint
    policy = LateFusionTransformer(
        input_dim=saved_cpt["model_arch"]["input_dim"],
        action_dim=saved_cpt["action_dim"],
        hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        pred_heads_arch=saved_cpt["model_arch"]["pred_heads_arch"],
        num_transformer_layers=saved_cpt["model_arch"][
            "num_transformer_layers"
        ],
    ).to(device)

    # Load the model parameters
    policy.load_state_dict(saved_cpt["parameters"])

    return policy.eval()


def rollout(env, policy, device):
    """Rollout policy in the environment."""

    # Storage
    goal_achieved = torch.zeros(env.num_worlds).to(device)
    collided = torch.zeros(env.num_worlds).to(device)
    off_road = torch.zeros(env.num_worlds).to(device)

    next_obs = env.reset()

    # Initialize masks
    live_agent_mask = env.cont_agent_mask.clone()

    for time_step in range(env.config.episode_len):

        # Get actions
        action, _, _, _ = policy(next_obs[live_agent_mask])

        action_template = torch.zeros(
            (env.num_worlds, env.max_agent_count), dtype=torch.int64
        ).to(device)
        action_template[live_agent_mask] = action

        # Step the environment
        env.step_dynamics(action_template)

        # Get infos from the environment
        next_obs = env.get_obs()
        dones = env.get_dones().bool()
        infos = env.get_infos()

        # Update mask
        live_agent_mask = torch.logical_or(live_agent_mask, ~dones)

        # Check terminal envs and store stats
        num_dones_per_world = (dones * env.cont_agent_mask).sum(dim=1)
        total_controlled_agents_per_world = env.cont_agent_mask.sum(dim=1)
        done_worlds = torch.where(
            (num_dones_per_world == total_controlled_agents_per_world)
        )[0]

        if len(done_worlds) > 0:
            for world in done_worlds:
                goal_achieved[world] = (
                    (infos.goal_achieved[world, :][env.cont_agent_mask[world]])
                    .sum()
                    .item()
                )
                collided[world] = (
                    infos.collided[world, :][env.cont_agent_mask[world]]
                    .sum()
                    .item()
                )
                off_road[world] = (
                    infos.off_road[world, :][env.cont_agent_mask[world]]
                    .sum()
                    .item()
                )

        # Check if all agents are done
        if dones.all():
            break

    # Aggregate metrics to get average ratio per scene
    goal_achieved = goal_achieved / env.cont_agent_mask.sum(dim=1)
    collided = collided / env.cont_agent_mask.sum(dim=1)
    off_road = off_road / env.cont_agent_mask.sum(dim=1)
    controlled_agents_in_world = env.cont_agent_mask.sum(dim=1)

    return goal_achieved, collided, off_road, controlled_agents_in_world


def evaluate_policy(
    env,
    policy,
    data_loader,
    dataset_name,
    store=False,
    store_dir="./",
    device="cuda",
):
    """Evaluate policy in the environment."""

    res_dict = {
        "scene": [],
        "goal_achieved": [],
        "collided": [],
        "off_road": [],
        "controlled_agents_in_world": [],
    }

    for batch in tqdm(
        data_loader,
        desc=f"Processing batches",
        total=len(data_loader),
        colour="blue",
    ):

        # Update simulator with the new batch of data
        env.reinit_scenarios(batch)

        # Rollout policy in the environments
        (
            goal_achieved,
            collided,
            off_road,
            controlled_agents_in_world,
        ) = rollout(env, policy, device)

        # Store results for the current batch
        scenario_names = [Path(path).stem for path in batch]
        res_dict["scene"].extend(scenario_names)
        res_dict["goal_achieved"].extend(goal_achieved.cpu().numpy())
        res_dict["collided"].extend(collided.cpu().numpy())
        res_dict["off_road"].extend(off_road.cpu().numpy())
        res_dict["controlled_agents_in_world"].extend(
            controlled_agents_in_world.cpu().numpy()
        )

    # Convert to pandas dataframe
    df_res = pd.DataFrame(res_dict)
    df_res["dataset"] = dataset_name

    if store:
        dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        df_res.to_csv(f"{store_dir}/{dataset_name}_{dt_string}.csv")

    return df_res


def load_config(cfg: str) -> Box:
    """Load configurations as a Box object.
    Args:
        cfg (str): Name of config file.

    Returns:
        Box: Box representation of configurations.
    """
    with open(f"{cfg}.yaml", "r") as stream:
        config = Box(yaml.safe_load(stream))
    return config


def make_env(config):
    """Make the environment with the given config."""

    scene_config = SceneConfig(
        path=config.data_dir,
        num_scenes=config.num_worlds,
        discipline=SelectionDiscipline.K_UNIQUE_N,
        k_unique_scenes=config.num_worlds,
        seed=config.sampling_seed,
    )

    # Override any default environment settings
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        norm_obs=config.normalize_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        dist_to_goal_threshold=config.dist_to_goal_threshold,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=config.use_lidar_obs,
        disable_classic_obs=True if config.use_lidar_obs else False,
        obs_radius=config.obs_radius,
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=config.max_controlled_agents,
        device=config.device,
    )

    return env


if __name__ == "__main__":

    # Make environment with the given config
    config = load_config("examples/experiments/eval/config/eval_base")
    env = make_env(config)

    # Make dataloaders
    train_loader = SceneDataLoader(
        root="data/processed/training",
        batch_size=100,  # Number of worlds
        dataset_size=1000,
        sample_with_replacement=False,
    )

    test_loader = SceneDataLoader(
        root="data/processed/testing",
        batch_size=100,
        dataset_size=1000,
        sample_with_replacement=False,
    )

    # Load policy
    policy = load_policy(
        path_to_cpt=config.model_path,
        device=config.device,
    )

    # Evaluate
    df_res_train = evaluate_policy(
        env=env,
        policy=policy,
        data_loader=train_loader,
        dataset_name="train",
        store=True,
        #store_dir="",
    )

    # df_res_test = evaluate_policy(
    #     env=env,
    #     policy=policy,
    #     data_loader=test_loader,
    #     dataset_name="test",
    #     store=True,
    # )
