import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data.types import Transitions
from stable_baselines3.common import policies

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from algorithms.il.data_generation import generate_state_action_pairs
from baselines.il.config import BehavCloningConfig


class CustomFeedForwardPolicy(policies.ActorCriticPolicy):
    """A feed forward policy network with a number of hidden units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=bc_config.net_arch)


def train_bc(
    env_config,
    scene_config,
    render_config,
    bc_config,
):
    rng = np.random.default_rng()

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=bc_config.max_cont_agents, 
        render_config=render_config,
        device=bc_config.device,
    )

    # Make custom policy
    policy = CustomFeedForwardPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
    )

    # Generate expert actions and observations
    (
        expert_obs,
        expert_actions,
        next_expert_obs,
        expert_dones,
    ) = generate_state_action_pairs(
        env=env,
        device=bc_config.device,
        discretize_actions=bc_config.discretize_actions,
        use_action_indices=bc_config.use_action_indices,
        make_video=bc_config.make_sanity_check_video,
    )

    # Convert to dataset of imitation "transitions"
    transitions = Transitions(
        obs=expert_obs.cpu().numpy(),
        acts=expert_actions.cpu().numpy(),
        infos=np.zeros_like(expert_dones.cpu()),  # Dummy
        next_obs=next_expert_obs.cpu().numpy(),
        dones=expert_dones.cpu().numpy().astype(bool),
    )

    # Define trainer
    bc_trainer = bc.BC(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        device=torch.device("cpu"),
    )

    # Train
    bc_trainer.train(
        n_epochs=bc_config.epochs,
        log_interval=bc_config.log_interval,
    )

    # Save policy
    if bc_config.save_model:
        bc_trainer.policy.save(
            path=f"{bc_config.model_path}/{bc_config.model_name}.pt"
        )


if __name__ == "__main__":
    
    NUM_WORLDS = 10

    # Configurations
    env_config = EnvConfig(use_bicycle_model=True)
    render_config = RenderConfig()
    scene_config = SceneConfig("data", NUM_WORLDS)
    bc_config = BehavCloningConfig()

    train_bc(
        env_config=env_config,
        render_config=render_config,
        scene_config=scene_config,
        bc_config=bc_config,
    )
