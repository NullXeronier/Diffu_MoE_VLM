"""
Crafter environment (Gymnasium API) and achievement metrics.

Crafter (Hafner, 2021) is a 2D open-world survival game with 64x64 pixel
observations, 17 discrete actions and 22 achievements such as EAT_PLANT,
MAKE_IRON_PICKAXE and COLLECT_DIAMOND. It is the environment family used in
"An Efficient Open World Environment for Multi-Agent Social Learning".
"""

from typing import Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

CRAFTER_ACTIONS: List[str] = [
    'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep',
    'place_stone', 'place_table', 'place_furnace', 'place_plant',
    'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
    'make_wood_sword', 'make_stone_sword', 'make_iron_sword',
]

CRAFTER_ACHIEVEMENTS: List[str] = [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling',
    'collect_stone', 'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow',
    'eat_plant', 'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe',
    'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
    'place_plant', 'place_stone', 'place_table', 'wake_up',
]


class CrafterEnv(gym.Env):
    """Gymnasium wrapper around `crafter.Env`"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 64, length: int = 10000, seed: Optional[int] = None, reward: bool = True):
        import crafter  # optional dependency: pip install -e ".[rl]"

        self._crafter = crafter
        self._kwargs = dict(size=(size, size), length=length, reward=reward)
        self._env = crafter.Env(seed=seed, **self._kwargs)
        self.observation_space = spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(CRAFTER_ACTIONS))
        self._last_obs = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            # Crafter seeds world generation at construction time
            self._env = self._crafter.Env(seed=seed, **self._kwargs)
        self._last_obs = self._env.reset()
        return self._last_obs, {'achievements': {name: 0 for name in CRAFTER_ACHIEVEMENTS}}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        self._last_obs = obs
        terminated = bool(done and info.get('discount', 1.0) == 0.0)  # the player died
        truncated = bool(done and not terminated)                     # episode length reached
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._last_obs


def crafter_score(success_rates: Dict[str, float]) -> float:
    """Crafter score: geometric mean of achievement success rates (in %), offset by 1"""
    rates = np.array([success_rates.get(name, 0.0) for name in CRAFTER_ACHIEVEMENTS])
    return float(np.exp(np.mean(np.log(1.0 + rates))) - 1.0)


class AchievementTracker:
    """Accumulates per-episode achievements, returns and lengths"""

    def __init__(self):
        self.episodes: List[Dict[str, int]] = []
        self.returns: List[float] = []
        self.lengths: List[int] = []

    def add_episode(self, achievements: Dict[str, int], episode_return: float = 0.0, length: int = 0):
        self.episodes.append({name: int(achievements.get(name, 0)) for name in CRAFTER_ACHIEVEMENTS})
        self.returns.append(float(episode_return))
        self.lengths.append(int(length))

    def extend(self, other: "AchievementTracker"):
        self.episodes += other.episodes
        self.returns += other.returns
        self.lengths += other.lengths

    def success_rates(self, last: Optional[int] = None) -> Dict[str, float]:
        """Percentage of episodes in which each achievement was unlocked at least once"""
        episodes = self.episodes[-last:] if last else self.episodes
        if not episodes:
            return {name: 0.0 for name in CRAFTER_ACHIEVEMENTS}
        return {
            name: 100.0 * sum(1 for ep in episodes if ep[name] > 0) / len(episodes)
            for name in CRAFTER_ACHIEVEMENTS
        }

    def summary(self, last: Optional[int] = None) -> Dict[str, float]:
        returns = self.returns[-last:] if last else self.returns
        lengths = self.lengths[-last:] if last else self.lengths
        rates = self.success_rates(last)
        out = {
            'episodes': len(returns),
            'return_mean': float(np.mean(returns)) if returns else 0.0,
            'length_mean': float(np.mean(lengths)) if lengths else 0.0,
            'score': crafter_score(rates),
        }
        out.update({f'achievements/{name}': rate for name, rate in rates.items()})
        return out


def format_achievements(rates: Dict[str, float], names: Optional[Iterable[str]] = None) -> str:
    names = list(names or CRAFTER_ACHIEVEMENTS)
    return ", ".join(f"{name.upper()}={rates[name]:.1f}%" for name in names)
