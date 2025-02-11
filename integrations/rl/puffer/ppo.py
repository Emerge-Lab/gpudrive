"""
This implementation is adapted from the demo in PufferLib by Joseph Suarez,
which in turn is adapted from Costa Huang's CleanRL PPO + LSTM implementation.
Links
- PufferLib: https://github.com/PufferAI/PufferLib/blob/dev/demo.py
- Cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

from pdb import set_trace as T
import numpy as np
import os
import random
import psutil
import time

from threading import Thread
from collections import defaultdict, deque

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

torch.set_float32_matmul_precision("high")

# Fast Cython GAE implementation
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from integrations.rl.puffer.c_gae import compute_gae
from integrations.rl.puffer.logging import print_dashboard, abbreviate


def create(config, vecenv, policy, optimizer=None, wandb=None):
    seed_everything(config.seed, config.torch_deterministic)
    profile = Profile()
    losses = make_losses()

    utilization = Utilization()
    msg = f"Model Size: {abbreviate(count_params(policy))} parameters"
    print_dashboard(
        config.env, utilization, 0, 0, profile, losses, {}, msg, clear=True
    )

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = vecenv.single_observation_space.dtype
    atn_shape = vecenv.single_action_space.shape
    total_agents = vecenv.num_agents

    lstm = policy.lstm if hasattr(policy, "lstm") else None

    # Rollout buffer
    experience = Experience(
        config.batch_size,
        config.bptt_horizon,
        config.minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        config.cpu_offload,
        config.device,
        lstm,
        total_agents,
    )

    uncompiled_policy = policy

    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(config.learning_rate), eps=1e-5
    )

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        experience=experience,
        profile=profile,
        losses=losses,
        wandb=wandb,
        global_step=0,
        global_step_pad=0,
        num_rollouts=0,
        resample_buffer=0,
        resample_counter=0,
        epoch=0,
        stats={},
        msg=msg,
        last_log_time=0,
        utilization=utilization,
    )


@pufferlib.utils.profile
def evaluate(data):

    # Resample data logic
    if data.config.resample_scenes:
        if (
            data.resample_counter < int(data.config.resample_limit)
            and data.config.resample_criterion == "global_step"
            and data.resample_buffer >= data.config.resample_interval
        ):
            print(f"Resampling scenarios: {data.resample_counter + 1:,}" + 
                (f" / {data.config.resample_limit}" if data.config.resample_limit is not None else ""))
            
            # Sample new batch of scenarios
            data.vecenv.resample_scenario_batch()
            data.resample_buffer = 0
            
            if data.config.resample_limit is not None:  # Increment counter only if there is a limit
                data.resample_counter += 1

    data.vecenv.wandb_obj = data.wandb
    data.vecenv.clear_render_storage()
    
    config, profile, experience = data.config, data.profile, data.experience

    with profile.eval_misc:
        policy = data.policy
        infos = defaultdict(list)
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    # Rollout loop
    while not experience.full:

        with profile.env:
            # Receive data from current timestep
            o, r, d, t, info, env_id, mask = data.vecenv.recv()
            env_id = env_id.tolist()

        with profile.eval_misc:
            data.global_step += sum(mask)
            data.global_step_pad += data.vecenv.total_agents
            data.resample_buffer += sum(mask)

            o = torch.as_tensor(o)
            o_device = o.to(config.device)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)

        with profile.eval_forward, torch.no_grad():
            if lstm_h is not None:
                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, _, value = policy(o_device)

            if config.device == "cuda":
                torch.cuda.synchronize()

        with profile.env:
            actions = actions.cpu().numpy()

            # Step the environment and reset if done
            data.vecenv.send(actions)

        with profile.eval_misc:
            value = value.flatten()
            mask = torch.as_tensor(mask)
            o = o if config.cpu_offload else o_device

            experience.store(o, value, actions, logprob, r, d, env_id, mask)

            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    infos[k].append(v)

    with profile.eval_misc:
        data.stats = {}

        for k, v in infos.items():
            try:
                data.stats[k] = np.mean(v)
                # Log variance for goal and collision metrics
                if "goal" in k or "collision" in k or "offroad" in k:
                    data.stats[f"{k}_std"] = np.std(v)
            except:
                continue

    data.num_rollouts += 1
    data.vecenv.global_step = data.global_step.copy()
    data.vecenv.iters = data.num_rollouts

    return data.stats, infos


@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_misc:
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np[idxs]
        values_np = experience.values_np[idxs]
        rewards_np = experience.rewards_np[idxs]
        # TODO: bootstrap between segment bounds
        advantages_np = compute_gae(
            dones_np, values_np, rewards_np, config.gamma, config.gae_lambda
        )
        experience.flatten_batch(advantages_np)

    # Optimizing the policy and value network
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb]
                obs = obs.to(config.device)
                atn = experience.b_actions[mb]
                log_probs = experience.b_logprobs[mb]
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb]

            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                        obs, state=lstm_state, action=atn
                    )
                    lstm_state = (
                        lstm_state[0].detach(),
                        lstm_state[1].detach(),
                    )
                else:
                    _, newlogprob, entropy, newvalue = data.policy(
                        obs.reshape(
                            -1, *data.vecenv.single_observation_space.shape
                        ),
                        action=atn,
                    )

                if config.device == "cuda":
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = (
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean()
                    )

                adv = adv.reshape(-1)
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - ret) ** 2
                    v_clipped = val + torch.clamp(
                        newvalue - val,
                        -config.vf_clip_coef,
                        config.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + v_loss * config.vf_coef
                )

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    data.policy.parameters(), config.max_grad_norm
                )
                data.optimizer.step()
                if config.device == "cuda":
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += (
                    pg_loss.item() / experience.num_minibatches
                )
                losses.value_loss += v_loss.item() / experience.num_minibatches
                losses.entropy += (
                    entropy_loss.item() / experience.num_minibatches
                )
                losses.old_approx_kl += (
                    old_approx_kl.item() / experience.num_minibatches
                )
                losses.approx_kl += (
                    approx_kl.item() / experience.num_minibatches
                )
                losses.clipfrac += clipfrac.item() / experience.num_minibatches

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = float(frac) * float(config.learning_rate)
            data.optimizer.param_groups[0]["lr"] = lrnow

        y_pred = experience.values_np
        y_true = experience.returns_np
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )
        losses.explained_variance = explained_var
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps

        # Logging
        if profile.update(data) or done_training:
            print_dashboard(
                config.env,
                data.utilization,
                data.global_step,
                data.epoch,
                profile,
                data.losses,
                data.stats,
                data.msg,
            )

            if (
                data.wandb is not None
                and data.global_step > 0
                and time.perf_counter() - data.last_log_time > 3.0
            ):
                data.last_log_time = time.perf_counter()
                data.wandb.log(
                    {
                        "performance/controlled_agent_sps": profile.controlled_agent_sps,
                        "performance/controlled_agent_sps_env": profile.controlled_agent_sps_env,
                        "performance/pad_agent_sps": profile.pad_agent_sps,
                        "performance/pad_agent_sps_env": profile.pad_agent_sps_env,
                        "performance/iters": data.num_rollouts,
                        "global_step": data.global_step,
                        "performance/epoch": data.epoch,
                        "performance/uptime": profile.uptime,
                        "train/learning_rate": data.optimizer.param_groups[0][
                            "lr"
                        ],
                        **{f"metrics/{k}": v for k, v in data.stats.items()},
                        **{f"train/{k}": v for k, v in data.losses.items()},
                    }
                )

        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f"Checkpoint saved at update {data.epoch}"


def close(data):
    data.vecenv.close()
    data.utilization.stop()
    config = data.config
    if data.wandb is not None:
        artifact_name = f"{config.exp_id}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()


class Profile:
    controlled_agent_sps: ... = 0
    controlled_agent_sps_env: ... = 0
    pad_agent_sps: ... = 0
    pad_agent_sps_env: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_misc_time: ... = 0

    def __init__(self):
        self.start = time.perf_counter()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.prev_steps = 0
        self.prev_steps_pad = 0
        self.prev_env_elapsed = 0

    def __iter__(self):
        yield "controlled_agent_sps", self.controlled_agent_sps
        yield "controlled_agent_sps_env", self.controlled_agent_sps_env
        yield "pad_agent_sps", self.pad_agent_sps
        yield "pad_agent_sps_env", self.pad_agent_sps_env
        yield "uptime", self.uptime
        yield "remaining", self.remaining
        yield "eval_time", self.eval_time
        yield "env_time", self.env_time
        yield "eval_forward_time", self.eval_forward_time
        yield "eval_misc_time", self.eval_misc_time
        yield "train_time", self.train_time
        yield "train_forward_time", self.train_forward_time
        yield "learn_time", self.learn_time
        yield "train_misc_time", self.train_misc_time

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, data, interval_s=1):
        global_step = data.global_step
        global_step_pad = data.global_step_pad
        if global_step == 0:
            return True

        uptime = time.perf_counter() - self.start
        if uptime - self.uptime < interval_s:
            return False

        # SPS = delta global step / delta time (s)
        self.controlled_agent_sps = (global_step - self.prev_steps) / (
            uptime - self.uptime
        )
        self.controlled_agent_sps_env = (global_step - self.prev_steps) / (
            self.env.elapsed - self.prev_env_elapsed
        )

        self.pad_agent_sps = (global_step_pad - self.prev_steps_pad) / (
            uptime - self.uptime
        )
        self.pad_agent_sps_env = (global_step_pad - self.prev_steps_pad) / (
            self.env.elapsed - self.prev_env_elapsed
        )

        self.prev_steps = global_step
        self.prev_steps_pad = global_step_pad
        self.prev_env_elapsed = self.env.elapsed
        self.uptime = uptime

        self.remaining = (
            data.config.total_timesteps - global_step
        ) / self.controlled_agent_sps
        self.eval_time = data._timers["evaluate"].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = data._timers["train"].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True


def make_losses():
    return pufferlib.namespace(
        policy_loss=0,
        value_loss=0,
        entropy=0,
        old_approx_kl=0,
        approx_kl=0,
        clipfrac=0,
        explained_variance=0,
    )


class Experience:
    """Flat tensor storage (buffer) and array views for faster indexing."""

    def __init__(
        self,
        batch_size,
        bptt_horizon,
        minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        cpu_offload=False,
        device="cuda",
        lstm=None,
        lstm_total_agents=0,
    ):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        pin = device == "cuda" and cpu_offload
        self.obs = torch.zeros(
            batch_size,
            *obs_shape,
            dtype=obs_dtype,
            pin_memory=pin,
            device=device if not pin else "cpu",
        )
        self.actions = torch.zeros(
            batch_size, *atn_shape, dtype=int, pin_memory=pin
        )
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)
        self.values = torch.zeros(batch_size, pin_memory=pin)

        # self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError("batch_size must be divisible by minibatch_size")

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError(
                "minibatch_size must be divisible by bptt_horizon"
            )

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, value, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[: self.batch_size - ptr]
        end = ptr + len(indices)

        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(
            sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__)
        )
        self.b_idxs_obs = (
            torch.as_tensor(
                idxs.reshape(
                    self.minibatch_rows,
                    self.num_minibatches,
                    self.bptt_horizon,
                ).transpose(1, 0, -1)
            )
            .to(self.obs.device)
            .long()
        )
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size
        )
        self.sort_keys = []
        self.ptr = 0
        self.step = 0
        return idxs

    def flatten_batch(self, advantages_np):
        advantages = torch.from_numpy(advantages_np).to(self.device)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_advantages = (
            advantages.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            )
            .transpose(0, 1)
            .reshape(self.num_minibatches, self.minibatch_size)
        )
        self.returns_np = advantages_np + self.values_np
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque(maxlen=maxlen)
        self.cpu_util = deque(maxlen=maxlen)
        self.gpu_util = deque(maxlen=maxlen)
        self.gpu_mem = deque(maxlen=maxlen)

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(mem.active / mem.total)
            self.gpu_util.append(torch.cuda.utilization())
            free, total = torch.cuda.mem_get_info()
            self.gpu_mem.append(free / total)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def save_checkpoint(data):

    config = data.config
    path = os.path.join(config.checkpoint_path, config.exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f"model_{data.epoch:06d}.pt"
    model_path = os.path.join(path, model_name)
    torch.save(data.uncompiled_policy.state_dict(), model_path)

    state = {
        "optimizer_state_dict": data.optimizer.state_dict(),
        "global_step": data.global_step,
        "agent_step": data.global_step,
        "update": data.epoch,
        "model_name": model_name,
        "exp_id": config.exp_id,
    }
    state_path = os.path.join(path, "trainer_state.pt")
    torch.save(state, state_path + ".tmp")
    os.rename(state_path + ".tmp", state_path)
    return model_path


def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic