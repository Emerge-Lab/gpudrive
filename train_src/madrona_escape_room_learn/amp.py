import torch
from typing import Optional
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass(init=False)
class AMPState:
    device_type: str
    enabled: bool
    compute_dtype: torch.dtype
    scaler: Optional[torch.cuda.amp.GradScaler]

    def __init__(self, dev, enable_mixed_precision):
        self.device_type = dev.type

        if enable_mixed_precision:
            self.enabled = True

            if dev.type == 'cuda':
                self.compute_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.compute_dtype = torch.bfloat16
                self.scaler = None
        else:
            self.enabled = False
            self.compute_dtype = torch.float32
            self.scaler = None

    @contextmanager
    def enable(self):
        if not self.enabled:
            try:
                yield
            finally:
                pass
        else:
            with torch.autocast(self.device_type, dtype=self.compute_dtype):
                try:
                    yield
                finally:
                    pass

    @contextmanager
    def disable(self):
        if not self.enabled:
            try:
                yield
            finally:
                pass
        else:
            with torch.autocast(self.device_type, enabled=False):
                try:
                    yield
                finally:
                    pass
