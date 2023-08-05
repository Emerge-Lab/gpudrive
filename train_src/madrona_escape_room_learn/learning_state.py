import torch
from dataclasses import dataclass
from typing import Optional

from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer

@dataclass
class LearningState:
    policy: ActorCritic
    optimizer : torch.optim.Optimizer
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
    value_normalizer: EMANormalizer
    amp: AMPState

    def save(self, update_idx, path):
        if self.scheduler != None:
            scheduler_state_dict = self.scheduler.state_dict()
        else:
            scheduler_state_dict = None

        if self.amp.scaler != None:
            scaler_state_dict = self.amp.scaler.state_dict()
        else:
            scaler_state_dict = None

        torch.save({
            'next_update': update_idx + 1,
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_state_dict,
            'value_normalizer': self.value_normalizer.state_dict(),
            'amp': {
                'device_type': self.amp.device_type,
                'enabled': self.amp.enabled,
                'compute_dtype': self.amp.compute_dtype,
                'scaler': scaler_state_dict,
            },
        }, path)

    def load(self, path):
        loaded = torch.load(path)

        self.policy.load_state_dict(loaded['policy'])
        self.optimizer.load_state_dict(loaded['optimizer'])

        if self.scheduler:
            self.scheduler.load_state_dict(loaded['scheduler'])
        else:
            assert(loaded['scheduler'] == None)

        self.value_normalizer.load_state_dict(loaded['value_normalizer'])

        amp_dict = loaded['amp']
        if self.amp.scaler:
            self.amp.scaler.load_state_dict(amp_dict['scaler'])
        else:
            assert(amp_dict['scaler'] == None)
        assert(
            self.amp.device_type == amp_dict['device_type'] and
            self.amp.enabled == amp_dict['enabled'] and
            self.amp.compute_dtype == amp_dict['compute_dtype'])

        return loaded['next_update']

    @staticmethod
    def load_policy_weights(path):
        loaded = torch.load(path)
        return loaded['policy']

