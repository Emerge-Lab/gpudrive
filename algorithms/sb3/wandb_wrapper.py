from collections import deque
from datetime import datetime
import os
import wandb

class WindowedCounter():
    def __init__(self, window_length, reduce_fn=sum):
        self.window = deque(maxlen=window_length)
        self.reduce_fn = reduce_fn

    def append(self, v):
        self.window.append(v)

    def value(self):
        return self.reduce_fn(self.window)

class NoWandbLogger():
    def __init__(self, exp_config, env_config):
        pass

    def define_metric(self, name, metric_fn):
        pass

    def log_defined_metrics_now(self):
        pass

    def log_histograms(self, name_to_raw_histogram):
        pass

    def log_video(self, **kwargs):

    def log_policy(self):
        assert False, "Model checkpointing requires enabling Wandb"

class WandbLogger():
    def __init__(self, exp_config, env_config):
        datetime_ = datetime.now().strftime("%m_%d_%H_%S")
        run_id = f"gpudrive_{datetime_}"
        self.run = wandb.init(
            project=exp_config.project_name,
            name=run_id,
            id=run_id,
            group=exp_config.group_name,
            sync_tensorboard=exp_config.sync_tensorboard,
            tags=exp_config.tags,
            mode=exp_config.wandb_mode,
            config={**exp_config.__dict__, **env_config.__dict__},
        )

        self.metric_name_to_fn = {}

    def define_metric(self, name, metric_fn):
        self.run.define_metric(name, step_metric="global_step")
        self.metric_name_to_fn[name] = metric_fn

    def log_defined_metrics(self):
        self.run.log({name: fn() for name, fn in self.metric_name_to_fn.items()})

    def log_histograms(self, name_to_raw_histogram):
        self.run.log({n: wandb.Histogram(h) for n,h in name_to_raw_histogram.items()})

    def log_video(self, num_timesteps, **kwargs):
        key = f"{render_type} | Global step: {num_timesteps:,}"
        self.run.log({key:kwargs})
            
    def log_policy(self, num_steps, **kwargs):
        

    # def log_video_once(self):

class Checkpointer():
    def __init__(self, wandb_logger, config):
        self.wandb_logger = wandb_logger
        self.save_policy_freq = config.save_policy_freq

        self.save_dir = os.path.join(wandb.run.dir, "policies")
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, num_timesteps, model):
        save_path = os.path.join(self.save_dir, f"policy_{num_timesteps}.zip")
        
        model.save(save_path)
        wandb_logger.log_policy(save_path, base_path=self.save_dir)

    def maybe_save(self, num_rollouts, num_timesteps, model):
        if num_rollouts % self.config.save_policy_freq:
            return

        self.save(num_timesteps, model)

class NoCheckpointer():
    def __init__(self, wandb_logger, config):
        pass

    def save(self, num_timesteps, model):
        pass
        
    def maybe_save(self, num_rollouts, num_timesteps, model):
        pass
