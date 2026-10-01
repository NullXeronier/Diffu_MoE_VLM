"""
Minimal vectorized environment with same-step auto-reset.

Each sub-environment runs in its own process (or in-process when
`num_workers=0`). When an episode ends, the worker records its return, length
and final achievements, resets, and returns the first observation of the next
episode, so rollouts never contain a terminal observation.
"""

import multiprocessing as mp
from typing import Callable, List, Optional, Tuple

import numpy as np


class _EpisodeEnv:
    """Wraps one env, tracks episode statistics and auto-resets"""

    def __init__(self, env_fn: Callable, seed: Optional[int]):
        self.env = env_fn()
        self.seed = seed
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None
        self.episode_return, self.episode_length = 0.0, 0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        episode = None
        done = terminated or truncated
        if done:
            episode = {
                'return': self.episode_return,
                'length': self.episode_length,
                'achievements': dict(info.get('achievements', {})),
                'terminated': terminated,
            }
            obs = self.reset()
        return obs, reward, done, episode


def _worker(remote, env_fn, seed):
    env = _EpisodeEnv(env_fn, seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'close':
                break
    finally:
        remote.close()


class VecEnv:
    def __init__(self, env_fns: List[Callable], seeds: Optional[List[int]] = None, num_workers: Optional[int] = None):
        self.num_envs = len(env_fns)
        seeds = seeds or [None] * self.num_envs
        self.parallel = (num_workers if num_workers is not None else self.num_envs) > 0
        if self.parallel:
            ctx = mp.get_context('spawn')
            self.remotes, self.processes = [], []
            for fn, seed in zip(env_fns, seeds):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker, args=(child, fn, seed), daemon=True)
                proc.start()
                child.close()
                self.remotes.append(parent)
                self.processes.append(proc)
        else:
            self.envs = [_EpisodeEnv(fn, seed) for fn, seed in zip(env_fns, seeds)]

    def reset(self) -> np.ndarray:
        if self.parallel:
            for r in self.remotes:
                r.send(('reset', None))
            return np.stack([r.recv() for r in self.remotes])
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[dict]]]:
        if self.parallel:
            for r, a in zip(self.remotes, actions):
                r.send(('step', int(a)))
            results = [r.recv() for r in self.remotes]
        else:
            results = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        obs, rewards, dones, episodes = zip(*results)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), list(episodes)

    def close(self):
        if self.parallel:
            for r in self.remotes:
                try:
                    r.send(('close', None))
                except (BrokenPipeError, EOFError):
                    pass
            for p in self.processes:
                p.join(timeout=5)
