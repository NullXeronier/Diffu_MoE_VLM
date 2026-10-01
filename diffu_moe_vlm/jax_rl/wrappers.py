"""Vectorized Craftax wrappers: batching, optimistic resets and episode logging"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct


class BatchEnv:
    """vmap reset/step over `num_envs` environments (env must auto-reset)"""

    def __init__(self, env, num_envs: int):
        self._env = env
        self.num_envs = num_envs

    def __getattr__(self, name):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        return jax.vmap(self._env.reset, in_axes=(0, None))(jax.random.split(key, self.num_envs), params)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._env.step, in_axes=(0, 0, 0, None))(keys, state, action, params)


class OptimisticResetBatchEnv:
    """
    Batched env where only `num_envs // reset_ratio` fresh worlds are generated
    per step and finished environments draw from that pool. World generation
    is the most expensive part of a Craftax step, so this is much faster than
    resetting every env every step. Wrap an env created with `auto_reset=False`.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int = 16):
        assert num_envs % reset_ratio == 0, "num_envs must be divisible by reset_ratio"
        self._env = env
        self.num_envs = num_envs
        self.num_resets = num_envs // reset_ratio

    def __getattr__(self, name):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        return jax.vmap(self._env.reset, in_axes=(0, None))(jax.random.split(key, self.num_envs), params)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        key_step, key_reset, key_pick = jax.random.split(key, 3)
        obs_st, state_st, reward, done, info = jax.vmap(self._env.step_env, in_axes=(0, 0, 0, None))(
            jax.random.split(key_step, self.num_envs), state, action, params)
        obs_re, state_re = jax.vmap(self._env.reset_env, in_axes=(0, None))(
            jax.random.split(key_reset, self.num_resets), params)
        # Each done env takes a fresh world; distinct done envs take distinct worlds when possible
        rank = jnp.cumsum(done) - 1
        pick = jnp.where(rank < self.num_resets, rank,
                         jax.random.randint(key_pick, (self.num_envs,), 0, self.num_resets))
        pick = jnp.clip(pick, 0, self.num_resets - 1)

        def select(fresh, stepped):
            fresh = fresh[pick]
            mask = done.reshape((-1,) + (1,) * (stepped.ndim - 1))
            return jnp.where(mask, fresh, stepped)

        state = jax.tree.map(select, state_re, state_st)
        obs = select(obs_re, obs_st)
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: chex.ArrayTree
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array
    timestep: jax.Array


class LogWrapper:
    """Tracks per-env episode return/length; adds them to `info` at episode end"""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        n = obs.shape[0]
        zeros_f, zeros_i = jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.int32)
        return obs, LogEnvState(env_state, zeros_f, zeros_i, zeros_f, zeros_i, zeros_i)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: LogEnvState, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_returns * (1 - done),
            episode_lengths=new_lengths * (1 - done),
            returned_episode_returns=jnp.where(done, new_returns, state.returned_episode_returns),
            returned_episode_lengths=jnp.where(done, new_lengths, state.returned_episode_lengths),
            timestep=state.timestep + 1,
        )
        info = dict(info)
        info["returned_episode"] = done
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        return obs, state, reward, done, info


def make_vec_env(env_name: str, num_envs: int, optimistic_resets: bool = True, reset_ratio: int = 16):
    from craftax.craftax_env import make_craftax_env_from_name

    if optimistic_resets:
        env = make_craftax_env_from_name(env_name, auto_reset=False)
        vec = OptimisticResetBatchEnv(env, num_envs, min(reset_ratio, num_envs))
    else:
        env = make_craftax_env_from_name(env_name, auto_reset=True)
        vec = BatchEnv(env, num_envs)
    return LogWrapper(vec), env.default_params
