import logging
import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy


class MultiAgentCallback(BaseCallback):
    """SB3 callback for gpudrive."""
    def __init__(
        self,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.wandb_run = wandb_run
    
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
        """
        ep_step = 90 - self.locals["env"].env.steps_remaining[0][0].item()
        wandb.define_metric("episode_step", step=ep_step)
        wandb.define_metric("episode/*", step_metric="episode_step")
        self.logger.record("episode/action", self.locals["clipped_actions"].item())
        self.logger.record("episode/reward", self.locals["rewards"].item())
        self.logger.record("episode/dist_to_goal", self.locals["new_obs"][0][-1].item())
        self.logger.record("episode/done", self.locals["dones"].item())
        
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        
        rewards = self.locals["rollout_buffer"].rewards.cpu().detach().numpy().flatten()
        observations = self.locals["rollout_buffer"].observations.cpu().detach().numpy()
        num_cont_agents = observations.shape[1]
        
        self.logger.record("rollout/global_step", self.num_timesteps)
        self.logger.record("rollout/avg_reward", np.mean(rewards))
        self.logger.record("rollout/std_reward", np.std(rewards))
        self.logger.record("rollout/avg_dist_to_goal", np.mean(observations[:, :, -1]))
        
    
    
    


