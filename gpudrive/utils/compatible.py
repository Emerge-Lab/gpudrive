from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import mediapy

from gpudrive.utils.config import load_config
from gpudrive.utils.rollout import rollout
from gpudrive.utils.env import make_env
from gpudrive.utils.checkpoint import load_policy

from gpudrive.env.dataset import SceneDataLoader

RESULTS_DIR = "examples/experimental/dataframes"


def get_compatibility_scores(
    env,
    agent,
    num_rollouts,
    device,
    identifier,
    deterministic=False,
    render_sim_state=False,
    render_every_t=10,
):
    """Evaluate policy in the environment."""

    if env.config.reward_type == "reward_conditioned":
        agent_weights = env.agent_type
        set_agent_type = True

    print(f"Controlling {env.max_cont_agents} agents per scenario")

    res_dict = {
        "scene": [],
        "goal_achieved_count": [],
        "goal_achieved_frac": [],
        "collided_count": [],
        "collided_frac": [],
        "off_road_count": [],
        "off_road_frac": [],
        "other_count": [],
        "other_frac": [],
        "num_controlled": [],
        "episode_lengths": [],
    }

    pbar = tqdm(
        range(num_rollouts),
        desc="Rollout",
        total=num_rollouts,
        colour="green",
    )

    for rollout_idx in pbar:
        pbar.set_description(
            f"Rollout {rollout_idx + 1}/{num_rollouts}, Batch size: {env.num_worlds}"
        )

        # Update simulator with the new batch of data
        env.swap_data_batch()

        # Rollout policy in the environments
        (
            goal_achieved_count,
            goal_achieved_frac,
            collided_count,
            collided_frac,
            off_road_count,
            off_road_frac,
            other_count,
            other_frac,
            controlled_agents_in_scene,
            sim_state_frames,
            agent_positions,
            episode_lengths,
        ) = rollout(
            env=env,
            policy=agent,
            device=device,
            deterministic=deterministic,
            render_sim_state=render_sim_state,
            set_agent_type=set_agent_type,
            agent_weights=agent_weights,
            render_every_n_steps=render_every_t,
        )

        # Get names from env
        scenario_to_worlds_dict = env.get_env_filenames()

        res_dict["scene"].extend(scenario_to_worlds_dict.values())
        res_dict["goal_achieved_count"].extend(
            goal_achieved_count.cpu().numpy()
        )
        res_dict["goal_achieved_frac"].extend(goal_achieved_frac.cpu().numpy())

        res_dict["collided_count"].extend(collided_count.cpu().numpy())
        res_dict["collided_frac"].extend(collided_frac.cpu().numpy())

        res_dict["off_road_count"].extend(off_road_count.cpu().numpy())
        res_dict["off_road_frac"].extend(off_road_frac.cpu().numpy())

        res_dict["other_count"].extend(other_count.cpu().numpy())
        res_dict["other_frac"].extend(other_frac.cpu().numpy())
        res_dict["num_controlled"].extend(
            controlled_agents_in_scene.cpu().numpy()
        )
        res_dict["episode_lengths"].extend(episode_lengths.cpu().numpy())

    # Convert to pandas dataframe
    df_res = pd.DataFrame(res_dict)
    df_res["Class"] = identifier

    print(f"\n Evaluated in {df_res.scene.nunique()} unique scenes.")

    return df_res, sim_state_frames, scenario_to_worlds_dict


if __name__ == "__main__":

    config = load_config(
        "/home/emerge/gpudrive/baselines/ppo/config/ppo_population"
    )
    EXP_ID = "rew_cond"

    # Set maximum number of agents that policy controls

    # Analysis settings
    config.data_dir = "data/processed/training"
    config.environment.num_worlds = 100
    config.dataset_size = 5000
    config.sample_with_replacement = True
    config.num_rollouts = 1
    config.environment.max_controlled_agents = 64  # 1
    config.device = "cuda"
    config.environment.reward_type = "reward_conditioned"
    config.environment.condition_mode = "fixed"
    config.environment.agent_type = torch.Tensor([-0.5, 1.0, -0.5])

    agent = load_policy(
        model_name="model_pop_play____R_10000__04_04_18_27_28_702_002500",
        path_to_cpt="/home/emerge/gpudrive/examples/experimental/models",
        env_config=config.environment,
        device=config.device,
    )

    # Create data loader
    train_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.environment.num_worlds,
        dataset_size=config.dataset_size,
        sample_with_replacement=True,
    )

    env = make_env(config.environment, train_loader, device=config.device)

    # Load sim agent trained through naive self-play
    # agent = NeuralNet.from_pretrained(
    #     "daphne-cornelisse/policy_S10_000_02_27"
    # ).to(config.device)

    df, frames, filenames = get_compatibility_scores(
        env=env,
        agent=agent,
        num_rollouts=config.num_rollouts,
        device=config.device,
        identifier=f"{EXP_ID}_{config.environment.max_controlled_agents}",
        render_sim_state=True,
        render_every_t=5,
    )
    df = df.dropna()

    df.to_csv(
        f"{RESULTS_DIR}/compatibility_scores_{config.environment.max_controlled_agents}.csv",
        index=False,
    )

    print(f"goal_achieved: {df['goal_achieved_frac'].mean()*100:.2f}")
    print(f"collided: {df['collided_frac'].mean()*100:.2f}")
    print(f"off_road: {df['off_road_frac'].mean()*100:.2f}")

    # Save videos
    sim_state_arrays = {k: np.array(v) for k, v in frames.items()}
    videos_dir = Path(f"videos/reward_conditioned")
    videos_dir.mkdir(parents=True, exist_ok=True)

    for env_id, frames in sim_state_arrays.items():

        filename = filenames[env_id]
        video_path = videos_dir / f"{filename}.gif"

        mediapy.write_video(
            str(video_path),
            frames,
            fps=5,
            codec="gif",
        )

        print(f"Saved video to {video_path}")
