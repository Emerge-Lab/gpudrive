from dataclasses import dataclass
from typing import Callable, List

import torch

@dataclass(frozen=True)
class PPOConfig:
    num_mini_batches: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    num_epochs: int = 1
    clip_value_loss: bool = False
    adaptive_entropy: bool = True

@dataclass(frozen=True)
class TrainConfig:
    num_updates: int
    steps_per_update: int
    num_bptt_chunks: int
    lr: float
    gamma: float
    ppo: PPOConfig
    gae_lambda: float = 1.0
    normalize_advantages: bool = True
    normalize_values : bool = True
    value_normalizer_decay : float = 0.99999
    mixed_precision : bool = False

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            if k == 'ppo':
                rep += f"\n  ppo:"
                for ppo_k, ppo_v in self.ppo.__dict__.items():
                    rep += f"\n    {ppo_k}: {ppo_v}"
            else:
                rep += f"\n  {k}: {v}" 

        return rep

@dataclass(frozen=True)
class SimInterface:
    step: Callable
    obs: List[torch.Tensor]
    actions: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
