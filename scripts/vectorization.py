from pdb import set_trace as T

import numpy as np
import gymnasium
from itertools import chain
import psutil
import time


from pufferlib import namespace
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.multi_env import create_precheck, GymnasiumMultiEnv, PettingZooMultiEnv
from pufferlib.exceptions import APIUsageError
import pufferlib.spaces
from gpudrive_gymnasium import GPUDriveEnv

RESET = 0
SEND = 1
RECV = 2

def setup(env_creator, env_args, env_kwargs):
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    driver_env = env_creator(*env_args, **env_kwargs)

    if isinstance(driver_env, GPUDriveEnv):
        multi_env_cls = GPUDriveEnv
        env_agents = driver_env.num_agents
        num_envs = driver_env.num_envs
        is_multiagent = True
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of GPUDriveEnv'
        )
    
    obs_space = _single_observation_space(driver_env)
    return driver_env, multi_env_cls, env_agents, num_envs

def _single_observation_space(env):
    return env.single_observation_space

def single_observation_space(state):
    return _single_observation_space(state.driver_env)

def _single_action_space(env):
    if isinstance(env, GymnasiumPufferEnv):
        return env.action_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_action_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_action_space(state):
    return _single_action_space(state.driver_env)

def structured_observation_space(state):
    return state.driver_env.structured_observation_space

def flat_observation_space(state):
    return state.driver_env.flat_observation_space

def unpack_batched_obs(state, obs):
    return state.driver_env.unpack_batched_obs(obs)

def recv_precheck(state):
    assert state.flag == RECV, 'Call reset before stepping'
    state.flag = SEND

def send_precheck(state):
    assert state.flag == SEND, 'Call reset + recv before send'
    state.flag = RECV

def reset_precheck(state):
    assert state.flag == RESET, 'Call reset only once on initialization'
    state.flag = RECV

def reset(self, seed=None):
    self.async_reset(seed)
    data = self.recv()
    return data[0], data[4]

def step(self, actions):
    self.send(actions)
    return self.recv()[:-1]

def aggregate_recvs(state, recvs):
    obs, rewards, dones, truncateds, infos, env_ids = list(zip(*recvs))
    assert all(state.workers_per_batch == len(e) for e in
        (obs, rewards, dones, truncateds, infos, env_ids))

    obs = np.concatenate(obs)
    rewards = np.concatenate(rewards)
    dones = np.concatenate(dones)
    truncateds = np.concatenate(truncateds)
    infos = [i for ii in infos for i in ii]
    
    obs_space = state.driver_env.structured_observation_space
    if isinstance(obs_space, pufferlib.spaces.Box):
        obs = obs.reshape(obs.shape[0], *obs_space.shape)

    # TODO: Masking will break for 1-agent PZ envs
    # Replace with check against is_multiagent (add it to state)
    if state.agents_per_env > 1:
        mask = [e['mask'] for ee in infos for e in ee.values()]
    else:
        mask = [e['mask'] for e in infos]

    env_ids = np.concatenate([np.arange( # Per-agent env indexing
        i*state.agents_per_worker, (i+1)*state.agents_per_worker) for i in env_ids])

    assert all(state.agents_per_batch == len(e) for e in
        (obs, rewards, dones, truncateds, env_ids, mask))
    assert len(infos) == state.envs_per_batch

    if state.mask_agents:
        return obs, rewards, dones, truncateds, infos, env_ids, mask

    return obs, rewards, dones, truncateds, infos, env_ids

def split_actions(state, actions, env_id=None):
    assert isinstance(actions, (list, np.ndarray))
    if type(actions) == list:
        actions = np.array(actions)

    assert len(actions) == state.agents_per_batch
    return np.array_split(actions, state.workers_per_batch)

def _unpack_shared_mem(shared_mem, n):
    np_buf = np.frombuffer(shared_mem.get_obj(), dtype=float)
    obs_arr = np_buf[:-3*n]
    rewards_arr = np_buf[-3*n:-2*n]
    terminals_arr = np_buf[-2*n:-n]
    truncated_arr = np_buf[-n:]

    return obs_arr, rewards_arr, terminals_arr, truncated_arr

def _worker_process(multi_env_cls, env_creator, env_args, env_kwargs,
        agents_per_env, envs_per_worker,
        worker_idx, shared_mem, send_pipe, recv_pipe):
    
    # I don't know if this helps. Sometimes it does, sometimes not.
    # Need to run more comprehensive tests
    #curr_process = psutil.Process()
    #curr_process.cpu_affinity([worker_idx])

    envs = multi_env_cls(env_creator, env_args, env_kwargs, n=envs_per_worker)
    obs_arr, rewards_arr, terminals_arr, truncated_arr = _unpack_shared_mem(
        shared_mem, agents_per_env * envs_per_worker)

    while True:
        request, args, kwargs = recv_pipe.recv()
        func = getattr(envs, request)
        response = func(*args, **kwargs)
        info = {}

        # TODO: Handle put/get
        if request in 'step reset'.split():
            obs, reward, done, truncated, info = response

            # TESTED: There is no overhead associated with 4 assignments to shared memory
            # vs. 4 assigns to an intermediate numpy array and then 1 assign to shared memory
            obs_arr[:] = obs.ravel()
            rewards_arr[:] = reward.ravel()
            terminals_arr[:] = done.ravel()
            truncated_arr[:] = truncated.ravel()

        send_pipe.send(info)


class CustomVectorized:
    '''
    A pufferlib wrapper around our GPUDrive environment which is already vectorized.

    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs

    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            mask_agents: bool = False,
            ) -> None:
        driver_env, multi_env_cls, agents_per_env, num_envs = setup(
            env_creator, env_args, env_kwargs)

        self.num_envs = num_envs
        self.agents_per_env = agents_per_env

        observation_size = int(np.prod(_single_observation_space(driver_env).shape))
        observation_dtype = _single_observation_space(driver_env).dtype

        self.observation_size = observation_size
        self.observation_dtype = observation_dtype
        self.driver_env = driver_env

        self.mask_agents = mask_agents

    def recv(self):
        recv_precheck(self)
        recvs = []
        next_env_id = []
        if self.env_pool:
            while len(recvs) < self.workers_per_batch:
                for key, _ in self.sel.select(timeout=None):
                    response_pipe = key.fileobj
                    env_id = self.recv_pipes.index(response_pipe)

                    if response_pipe.poll():
                        info = response_pipe.recv()
                        o, r, d, t = _unpack_shared_mem(
                            self.shared_mem[env_id], self.agents_per_env * self.envs_per_worker)
                        o = o.reshape(
                            self.agents_per_env*self.envs_per_worker,
                            self.observation_size).astype(self.observation_dtype)

                        recvs.append((o, r, d, t, info, env_id))
                        next_env_id.append(env_id)

                    if len(recvs) == self.workers_per_batch:                    
                        break
        else:
            for env_id in range(self.workers_per_batch):
                response_pipe = self.recv_pipes[env_id]
                info = response_pipe.recv()
                o, r, d, t = _unpack_shared_mem(
                    self.shared_mem[env_id], self.agents_per_env * self.envs_per_worker)
                o = o.reshape(
                    self.agents_per_env*self.envs_per_worker,
                    self.observation_size).astype(self.observation_dtype)

                recvs.append((o, r, d, t, info, env_id))
                next_env_id.append(env_id)

        self.prev_env_id = next_env_id
        return aggregate_recvs(self, recvs)

    def send(self, actions):
        send_precheck(self)
        actions = split_actions(self, actions)
        for i, atns in zip(self.prev_env_id, actions):
            self.send_pipes[i].send(("step", [atns], {}))

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            for pipe in self.send_pipes:
                pipe.send(("reset", [], {}))
        else:
            for idx, pipe in enumerate(self.send_pipes):
                pipe.send(("reset", [], {"seed": seed+idx}))

    def put(self, *args, **kwargs):
        # TODO: Update this
        for queue in self.request_queues:
            queue.put(("put", args, kwargs))

    def get(self, *args, **kwargs):
        # TODO: Update this
        for queue in self.request_queues:
            queue.put(("get", args, kwargs))

        idx = -1
        recvs = []
        while len(recvs) < self.workers_per_batch // self.envs_per_worker:
            idx = (idx + 1) % self.num_workers
            queue = self.response_queues[idx]

            if queue.empty():
                continue

            response = queue.get()
            if response is not None:
                recvs.append(response)

        return recvs

    def close(self):
        for pipe in self.send_pipes:
            pipe.send(("close", [], {}))

        for p in self.processes:
            p.terminate()

        for p in self.processes:
            p.join()