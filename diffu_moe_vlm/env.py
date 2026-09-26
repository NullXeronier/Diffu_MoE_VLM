"""
Symbolic Minecraft tech-tree environment (Gymnasium API).

Actions are high-level (verb, item) macros such as ("mine", "wood") or
("craft", "stick"). Their effect on the inventory follows the tech tree in
`core`, so a task succeeds only when the agent actually gathers and crafts
the required items. There is no 3D world: the RGB observation is a blank
frame kept for interface compatibility with pixel-based policies.
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import apply_action, mc, obtain_method, task_registry

ActionLike = Union[int, np.integer, Tuple[str, str], Dict[str, Any]]


class MinecraftGymnasiumEnv(gym.Env):
    """Tech-tree crafting environment with a Gymnasium interface"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 name: str = "Plains",
                 img_size: Tuple[int, int] = (640, 480),
                 rgb_only: bool = False,
                 max_steps: int = 1000,
                 action_failure_prob: float = 0.0,
                 **kwargs):
        """
        Args:
            name: biome name (informational only)
            img_size: (width, height) of the RGB observation
            rgb_only: omit the depth observation
            max_steps: episode length before truncation
            action_failure_prob: probability that a valid action randomly fails
                (models imperfect low-level control; 0 = deterministic)
        """
        super().__init__()

        self.name = name
        self.img_size = tuple(img_size)
        self.rgb_only = rgb_only
        self._max_steps = max_steps
        self.action_failure_prob = action_failure_prob

        self.items = list(mc.ALL_ITEMS)
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.macro_actions = [("noop", "")] + [(obtain_method(item), item) for item in self.items]
        self.macro_index = {a: i for i, a in enumerate(self.macro_actions)}

        self.action_space = spaces.Discrete(len(self.macro_actions))
        self.observation_space = self._create_observation_space()

        self._current_step = 0
        self._inventory: Dict[str, int] = {}
        self._target: Optional[str] = None
        self._target_quantity = 1

    def _create_observation_space(self):
        width, height = self.img_size
        obs_space = {
            'rgb': spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
            'inventory': spaces.Box(low=0, high=np.iinfo(np.int32).max,
                                    shape=(len(self.items),), dtype=np.int32),
        }
        if not self.rgb_only:
            obs_space['depth'] = spaces.Box(low=0, high=255, shape=(height, width), dtype=np.uint8)
        return spaces.Dict(obs_space)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Options:
            task: task name from the registry (e.g. "obtain_wooden_slab")
            target / target_quantity: explicit target item instead of a task
            initial_inventory: starting inventory (default empty)
        """
        super().reset(seed=seed)
        options = options or {}

        self._current_step = 0
        self._inventory = {k: v for k, v in options.get('initial_inventory', {}).items() if v > 0}
        self._target = options.get('target')
        if self._target is None and options.get('task'):
            task = task_registry.get_task(options['task'])
            if task is None:
                raise ValueError(f"Unknown task: {options['task']}")
            self._target = task['target']
        self._target_quantity = options.get('target_quantity', 1)

        return self._get_observation(), self._get_info(action_success=True, action_error="")

    def step(self, action: ActionLike):
        self._current_step += 1
        verb, item = self.decode_action(action)

        success, error = True, ""
        if verb != "noop":
            new_inventory, success, error = apply_action(self._inventory, verb, item)
            if success and self.action_failure_prob > 0 and self.np_random.random() < self.action_failure_prob:
                success, error = False, f"{verb} {item} failed (control failure)"
            if success:
                self._inventory = new_inventory

        task_success = self._task_success()
        reward = 1.0 if task_success else 0.0
        terminated = task_success
        truncated = not terminated and self._current_step >= self._max_steps

        info = self._get_info(action_success=success, action_error=error)
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        return self._get_observation()['rgb']

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def decode_action(self, action: ActionLike) -> Tuple[str, str]:
        """Accept a macro index, a (verb, item) tuple or a {'type', 'item'} dict"""
        if isinstance(action, (int, np.integer)):
            return self.macro_actions[int(action)]
        if isinstance(action, tuple):
            return action
        if isinstance(action, dict) and 'type' in action:
            return action['type'], action.get('item', "")
        # Legacy low-level actions (movement/camera/attack) have no symbolic effect
        return "noop", ""

    def encode_action(self, verb: str, item: str) -> int:
        return self.macro_index[(verb, item)]

    def _task_success(self) -> bool:
        return self._target is not None and self._inventory.get(self._target, 0) >= self._target_quantity

    def _get_observation(self):
        width, height = self.img_size
        inventory = np.zeros(len(self.items), dtype=np.int32)
        for item, count in self._inventory.items():
            if item in self.item_index:
                inventory[self.item_index[item]] = count
        obs = {'rgb': np.zeros((height, width, 3), dtype=np.uint8), 'inventory': inventory}
        if not self.rgb_only:
            obs['depth'] = np.zeros((height, width), dtype=np.uint8)
        return obs

    def _get_info(self, action_success: bool, action_error: str) -> Dict[str, Any]:
        return {
            'step': self._current_step,
            'inventory': dict(self._inventory),
            'target': self._target,
            'action_success': action_success,
            'action_error': action_error,
            'task_success': self._task_success(),
        }


class MinecraftEnvRegistry:
    """Registry for different Minecraft environments"""

    _environments = {
        'Plains': MinecraftGymnasiumEnv,
        'Forest': MinecraftGymnasiumEnv,
        'Mountain': MinecraftGymnasiumEnv,
    }

    @classmethod
    def make(cls, name: str, **kwargs):
        if name not in cls._environments:
            raise ValueError(f"Environment {name} not found")
        return cls._environments[name](name=name, **kwargs)

    @classmethod
    def register(cls, name: str, env_class):
        cls._environments[name] = env_class


def MineDojoEnv(name: str = "Plains", **kwargs):
    """Compatibility function to replace the MineDojo environment"""
    return MinecraftEnvRegistry.make(name, **kwargs)
