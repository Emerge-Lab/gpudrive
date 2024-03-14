import random
import sys

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from gymnasium import spaces
import numpy as np
from torch import nn
from torch.distributions import Categorical
import torch as th

from callbacks import MetricsCallback
from sb3_wrapper import SB3Wrapper
from ppo.ppo import PPO


# TODO(ev) move into util
def set_seed_everywhere(seed):
    # set the random seed for torch, numpy, and python
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# temporary policy just to get things running
class CustomModule(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete, lr_schedule, **kwargs):
        super().__init__()
        n_input_channels = observation_space.shape[-1]
        self.ff = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # TODO(ev) make the NN configurable
        self.action_linear = nn.Linear(256, action_space.n)
        self.value_linear = nn.Linear(256, 1)
        self.normalizer = 100

        # TODO(ev) actually take the optimizer kwargs from sb3
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1), eps=1e-5)

    def __call__(self, observations: th.Tensor) -> th.Tensor:
        out = self.ff(self.normalize_obs(observations))
        action_logits = self.action_linear(out)
        values = self.value_linear(out)
        dist = Categorical(
            logits=action_logits,
        )
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.int(), values, log_probs
    
    def normalize_obs(self, observation):
        return observation / self.normalizer

    def evaluate_actions(self, observations: th.Tensor, actions: th.Tensor):
        # TODO(ev) code duplication
        out = self.ff(self.normalize_obs(observations))
        action_logits = self.action_linear(out)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        values = self.value_linear(out)
        entropy = dist.entropy()
        return values, log_probs, entropy

    def predict_values(self, observations: th.Tensor) -> th.Tensor:
        return self.value_linear(self.ff(self.normalize_obs(observations)))

    def set_training_mode(self, mode: bool):
        self.train(mode)


def get_config(config_path, config_name, overrides):
    if overrides is None:
        overrides = []
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    return compose(config_name=config_name, overrides=overrides, return_hydra_config=True)


if __name__ == "__main__":
    config = get_config("./", "config", sys.argv[1:])
    ppo_config = config.runner_configs

    set_seed_everywhere(config.seed)
    env = SB3Wrapper(config)

    model = PPO(CustomModule, env, **dict(ppo_config))

    callback_list = []
    callback_list += [MetricsCallback(verbose=1)]
    # if config.render_callback:
    #     callback_list += [VideoCallback(verbose=1, video_freq=100)]

    model.learn(10000000000, callback=callback_list)