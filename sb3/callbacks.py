from copy import deepcopy
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gpudrive_gym_env import Env


class MetricsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # TODO(ev) the getattr method in training_env is not working so we need the _env for now
        stats = self.training_env._env.metrics.get_stats()
        for k, v in stats.items():
            self.logger.record(k, v)

    def _on_training_end(self) -> None:
        pass
