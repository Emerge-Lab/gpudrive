import pandas as pd
import torch
from time import perf_counter

from pygpudrive.env.config import EnvConfig
from pygpudrive.env.base_environment import Env


if __name__ == "__main__":

    speed_config = {
        "num_steps": 1000,
        "max_controlled_agents": 50,
    }

    df = pd.dataframe()

    for num_worlds in [1, 10]:

        env_config = EnvConfig(
            ego_state=True,
            road_map_obs=False,
            partner_obs=False,
            normalize_obs=False,
        )

        # Initialize the environment
        env = Env(
            config=env_config,
            num_worlds=1,
            max_cont_agents=50,
        )

        # Take steps in the environment
        env.reset()
        for num_worlds in range(1000):

            # Sample random actions
            rand_actions = torch.tensor(
                [env.action_space.sample() for _ in range(50 * num_worlds)]
            ).reshape(num_worlds, 50)

            env.step(rand_actions)
        env.close()
