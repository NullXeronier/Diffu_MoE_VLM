"""
Policy runners, evaluation and demonstration collection.

A runner maps a batch of observations (N, H, W, 3) uint8 to actions (N,)
and is told when episodes end so chunked policies can drop queued actions.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..crafter_env import AchievementTracker
from .vec_env import VecEnv


class RandomRunner:
    def __init__(self, num_actions: int, seed: int = 0):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(self.num_actions, size=len(obs))

    def episode_done(self, env_index: int):
        pass


class ActorCriticRunner:
    def __init__(self, model, device: str = 'cpu', deterministic: bool = False):
        self.model, self.device, self.deterministic = model, device, deterministic

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _, _ = self.model.act(torch.as_tensor(obs, device=self.device), deterministic=self.deterministic)
        return action.cpu().numpy()

    def episode_done(self, env_index: int):
        pass


class DiffusionRunner:
    """Receding-horizon execution: sample an action chunk, execute the first `execute_steps`, replan"""

    def __init__(self, model, device: str = 'cpu', execute_steps: int = 4, sample_steps: Optional[int] = 10):
        self.model, self.device = model, device
        self.execute_steps = execute_steps
        self.sample_steps = sample_steps
        self.queues: Dict[int, List[int]] = {}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        need = [i for i in range(len(obs)) if not self.queues.get(i)]
        if need:
            chunks = self.model.sample(torch.as_tensor(obs[need], device=self.device), num_steps=self.sample_steps)
            for i, chunk in zip(need, chunks.cpu().tolist()):
                self.queues[i] = chunk[:self.execute_steps]
        return np.array([self.queues[i].pop(0) for i in range(len(obs))])

    def episode_done(self, env_index: int):
        self.queues[env_index] = []


@torch.no_grad()
def evaluate(runner, envs: VecEnv, num_episodes: int, max_steps: int = 1_000_000) -> AchievementTracker:
    """Run until `num_episodes` episodes have finished across all envs"""
    tracker = AchievementTracker()
    obs = envs.reset()
    steps = 0
    while len(tracker.episodes) < num_episodes and steps < max_steps:
        obs, _, _, episodes = envs.step(runner(obs))
        steps += 1
        for i, ep in enumerate(episodes):
            if ep is not None:
                runner.episode_done(i)
                if len(tracker.episodes) < num_episodes:
                    tracker.add_episode(ep['achievements'], ep['return'], ep['length'])
    return tracker


@torch.no_grad()
def collect_demonstrations(runner, envs: VecEnv, num_steps: int, path=None) -> Dict[str, np.ndarray]:
    """
    Record (observation, action) pairs. Returns arrays `obs` (N, H, W, 3),
    `actions` (N,) and `episode` (N,) ids; frames of one episode are contiguous.
    """
    obs = envs.reset()
    per_env = [{'obs': [], 'actions': []} for _ in range(envs.num_envs)]
    out_obs, out_actions, out_episode = [], [], []
    episode_id = 0
    total = 0

    def flush(i):
        nonlocal episode_id
        if per_env[i]['actions']:
            out_obs.extend(per_env[i]['obs'])
            out_actions.extend(per_env[i]['actions'])
            out_episode.extend([episode_id] * len(per_env[i]['actions']))
            episode_id += 1
        per_env[i] = {'obs': [], 'actions': []}

    while total < num_steps:
        actions = runner(obs)
        for i in range(envs.num_envs):
            per_env[i]['obs'].append(obs[i])
            per_env[i]['actions'].append(int(actions[i]))
        obs, _, _, episodes = envs.step(actions)
        total += envs.num_envs
        for i, ep in enumerate(episodes):
            if ep is not None:
                runner.episode_done(i)
                flush(i)
    for i in range(envs.num_envs):
        flush(i)

    data = {
        'obs': np.stack(out_obs).astype(np.uint8),
        'actions': np.array(out_actions, dtype=np.int64),
        'episode': np.array(out_episode, dtype=np.int64),
    }
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
    return data


class ActionChunkDataset(torch.utils.data.Dataset):
    """(obs_t, actions_{t..t+H-1}) windows that stay within one episode"""

    def __init__(self, data: Dict[str, np.ndarray], horizon: int):
        self.obs = data['obs']
        self.actions = data['actions']
        episode = data['episode']
        ends_ok = np.zeros(len(episode), dtype=bool)
        if len(episode) >= horizon:
            same = episode[:len(episode) - horizon + 1] == episode[horizon - 1:]
            ends_ok[:len(same)] = same
        self.starts = np.nonzero(ends_ok)[0]
        self.horizon = horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return torch.as_tensor(self.obs[s]), torch.as_tensor(self.actions[s:s + self.horizon])
