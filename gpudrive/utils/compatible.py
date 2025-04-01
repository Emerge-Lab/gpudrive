from tqdm import tqdm
import torch
import pandas as pd

from gpudrive.utils.config import load_config
from gpudrive.utils.rollout import rollout
from gpudrive.utils.env import make
from gpudrive.utils.checkpoint import make_agent

RESULTS_DIR = "ada/eval/notebooks/results"


def get_compatibility_scores(
    env,
    agent,
    data_loader,
    num_rollouts,
    device,
    identifier,
    deterministic=False,
    render_sim_state=False,
):
    """Evaluate policy in the environment."""

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

    return df_res


if __name__ == "__main__":
    # config = load_config("config/ada/eval/sp_cross_play")
    config = load_config("config/exp/pop_play_rew_cond_flat")
    EXP_ID = "rew_cond"

    # Set maximum number of agents that policy controls

    # Analysis settings
    config.data_dir = "/home/emerge/gpudrive/data/processed/training"
    config.num_worlds = 100
    config.dataset_size = 500
    config.sample_with_replacement = True
    config.num_rollouts = 10
    config.max_controlled_agents = 64
    config.condition_mode = "fixed"
    config.device = "cuda"
    # config.agent_type = torch.Tensor([-0.75, 1.0, -0.75])

    env = make(config)

    # Load sim agent trained through naive self-play
    POLICY = "models/baselines/model_pop_play_rew_cond___S_500__03_16_10_41_19_391_007522.pt"
    print(f"Loading policy: {POLICY} \n")
    cpt = torch.load(POLICY, map_location=config.device, weights_only=False)
    agent = make_agent(cpt, config, device=config.device)
    # Load sim agent trained through naive self-play
    # agent = NeuralNet.from_pretrained(
    #     "daphne-cornelisse/policy_S10_000_02_27"
    # ).to(config.device)

    df = get_compatibility_scores(
        env=env,
        agent=agent,
        data_loader=env.data_loader,
        num_rollouts=config.num_rollouts,
        device=config.device,
        identifier=f"{EXP_ID}_{config.max_controlled_agents}",
    )
    df = df.dropna()

    df.to_csv(
        f"{RESULTS_DIR}/compatibility_scores_{config.max_controlled_agents}.csv",
        index=False,
    )

    print(f"goal_achieved: {df['goal_achieved_frac'].mean()*100:.2f}")
    print(f"collided: {df['collided_frac'].mean()*100:.2f}")
    print(f"off_road: {df['off_road_frac'].mean()*100:.2f}")