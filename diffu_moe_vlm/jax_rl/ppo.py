"""
PPO / PPO-RNN for Craftax in pure JAX (after PureJaxRL and the Craftax baselines).

`make_train(config)` returns a function `train(rng)` that runs the whole
training loop under `jax.lax.scan`, so it can be jitted end to end. Options:
    rnn   GRU memory (PPO-RNN); otherwise the feed-forward baseline network
    moe   top-k mixture-of-experts hidden layers (+ load-balancing loss)
    icm   Intrinsic Curiosity Module bonus: icm_reward_coef * error / scale, where
          `icm_normalize` picks the scale (see ICM_NORMALIZERS):
            mean  running mean of all forward-model errors so far (default).
                  The first batch averages exactly 1; later batches average
                  (current mean error / all-time mean error), and single steps
                  are unbounded. It sets the typical scale, not a hard limit
            ema   exponential moving mean (`icm_ema_decay` per update), which
                  follows the error as the forward model improves
            std   running standard deviation. Not enough on its own: the
                  error's mean is much larger than its spread, so the bonus
                  stays ~10x too large
            none  raw error; the unscaled bonus is what made the reference ICM
                  run collapse to ~0 extrinsic reward
          The icm/* metrics track the raw error, the bonus and its share of the
          total reward, to detect a runaway bonus (run_icm_ablation.py).

Per update, metrics (episode return/length and achievement success rates of
episodes that finished in that update, losses) are returned stacked and, if
`log_fn` is given, streamed to Python with `jax.debug.callback`.
"""

import os
from typing import Any, Callable, Dict, NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from .networks import ICM, ActorCritic, ActorCriticRNN, ScannedGRU
from .wrappers import make_vec_env

DEFAULT_CONFIG: Dict[str, Any] = {
    # Values follow the Craftax PPO baselines; check them against the run you want to reproduce
    "env_name": "Craftax-Symbolic-v1",
    "total_timesteps": 1_000_000_000,
    "num_envs": 1024,
    "num_steps": 64,
    "update_epochs": 4,
    "num_minibatches": 8,
    "lr": 2e-4,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.8,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    "layer_size": 512,
    "num_layers": 3,
    "activation": "tanh",
    "rnn": False,
    "moe": False,
    "num_experts": 4,
    "top_k": 2,
    "moe_aux_coef": 0.01,
    "optimistic_resets": True,
    "reset_ratio": 16,
    "icm": False,
    "icm_reward_coef": 0.01,
    "icm_lr": 3e-4,
    "icm_forward_coef": 1.0,
    "icm_inverse_coef": 1.0,
    "icm_normalize": "mean",
    "icm_ema_decay": 0.99,
}

ICM_NORMALIZERS = ("none", "std", "mean", "ema")


class Transition(NamedTuple):
    reset: jax.Array      # episode ended before this observation (resets the RNN state)
    done: jax.Array       # episode ended after this step
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: Dict[str, jax.Array]


class RunningMoments(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def update_moments(m: RunningMoments, x: jax.Array) -> RunningMoments:
    """Parallel (Chan et al.) update of running mean/variance with a batch"""
    b_mean, b_var, b_count = x.mean(), x.var(), x.size
    delta = b_mean - m.mean
    total = m.count + b_count
    mean = m.mean + delta * b_count / total
    m2 = m.var * m.count + b_var * b_count + delta ** 2 * m.count * b_count / total
    return RunningMoments(mean, m2 / total, total)


def update_ema(m: RunningMoments, x: jax.Array, decay: float) -> RunningMoments:
    """Exponential moving mean of batch means (`count` = number of batches; the first batch initializes it)"""
    b_mean = x.mean()
    mean = jnp.where(m.count < 1, b_mean, decay * m.mean + (1.0 - decay) * b_mean)
    return RunningMoments(mean, m.var, jnp.floor(m.count) + 1)


def icm_normalizer(config) -> str:
    """`icm_normalize` as a mode name; booleans from older configs map to mean / none"""
    mode = config["icm_normalize"]
    if isinstance(mode, bool):
        mode = "mean" if mode else "none"
    if mode not in ICM_NORMALIZERS:
        raise ValueError(f"icm_normalize must be one of {ICM_NORMALIZERS}, got {mode!r}")
    return mode


def craftax_score(rates: jax.Array) -> jax.Array:
    """Crafter/Craftax score: geometric mean of success rates in %, exp(mean(ln(1 + s))) - 1"""
    return jnp.exp(jnp.log1p(rates).mean()) - 1.0


def compute_gae(rewards, values, dones, last_value, gamma: float, gae_lambda: float):
    """Generalized advantage estimation over (T, B) arrays; `dones[t]` ends the episode after step t"""
    def step(carry, x):
        gae, next_value = carry
        reward, value, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value), (rewards, values, dones),
                                 reverse=True)
    return advantages, advantages + values


def build_network(config, action_dim: int) -> nn.Module:
    common = dict(action_dim=action_dim, width=config["layer_size"], moe=config["moe"],
                  num_experts=config["num_experts"], top_k=config["top_k"])
    if config["rnn"]:
        return ActorCriticRNN(activation="relu" if config["activation"] == "tanh" else config["activation"], **common)
    return ActorCritic(num_layers=config["num_layers"], activation=config["activation"], **common)


def make_train(config: Dict[str, Any], log_fn: Optional[Callable[[Dict, int], None]] = None):
    config = {**DEFAULT_CONFIG, **config}
    env, env_params = make_vec_env(config["env_name"], config["num_envs"],
                                   config["optimistic_resets"], config["reset_ratio"])
    num_envs, num_steps = config["num_envs"], config["num_steps"]
    num_updates = config["total_timesteps"] // num_steps // num_envs
    assert num_updates > 0, "total_timesteps must be at least num_envs * num_steps"
    assert num_envs % config["num_minibatches"] == 0
    action_dim = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    network = build_network(config, action_dim)
    rnn = config["rnn"]
    icm_net = ICM(action_dim) if config["icm"] else None
    icm_mode = icm_normalizer(config)

    def lr_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / num_updates
        return config["lr"] * frac

    def apply_net(params, hstate, obs, reset):
        """obs/reset are (T, B, ...) for the RNN and (..., obs_dim) for the feed-forward net"""
        if rnn:
            return network.apply(params, hstate, (obs, reset))
        pi, value, aux = network.apply(params, obs)
        return hstate, pi, value, aux

    def init_runner(rng):
        rng, k_net, k_icm, k_env = jax.random.split(rng, 4)
        init_h = ScannedGRU.initialize_carry(num_envs, config["layer_size"])
        if rnn:
            params = network.init(k_net, init_h, (jnp.zeros((1, num_envs, obs_dim)), jnp.zeros((1, num_envs), bool)))
        else:
            params = network.init(k_net, jnp.zeros((1, obs_dim)))
        lr = lr_schedule if config["anneal_lr"] else config["lr"]
        tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        if icm_net is not None:
            icm_params = icm_net.init(k_icm, jnp.zeros((1, obs_dim)), jnp.zeros((1, obs_dim)), jnp.zeros((1,), jnp.int32))
            icm_state = TrainState.create(apply_fn=icm_net.apply, params=icm_params,
                                          tx=optax.adam(config["icm_lr"]))
        else:
            icm_state = None
        moments = RunningMoments(jnp.zeros(()), jnp.ones(()), jnp.asarray(1e-4))

        obs, env_state = env.reset(k_env, env_params)
        return (train_state, icm_state, moments, env_state, obs, jnp.zeros(num_envs, bool), init_h, rng)

    def update_step(runner, update_idx):
        train_state, icm_state, moments, env_state, last_obs, last_done, hstate, rng = runner
        init_hstate = hstate

        # ---- rollout ----
        def env_step(carry, _):
            train_state, env_state, last_obs, last_done, hstate, rng = carry
            rng, k_act, k_env = jax.random.split(rng, 3)
            if rnn:
                hstate, pi, value, _ = apply_net(train_state.params, hstate, last_obs[None], last_done[None])
                pi_logits, value = pi.logits[0], value[0]
                pi = type(pi)(logits=pi_logits)
            else:
                _, pi, value, _ = apply_net(train_state.params, None, last_obs, None)
            action = pi.sample(seed=k_act)
            log_prob = pi.log_prob(action)
            obs, env_state, reward, done, info = env.step(k_env, env_state, action, env_params)
            t = Transition(last_done, done, action, value, reward, log_prob, last_obs, info)
            return (train_state, env_state, obs, done, hstate, rng), t

        (train_state, env_state, last_obs, last_done, hstate, rng), traj = jax.lax.scan(
            env_step, (train_state, env_state, last_obs, last_done, hstate, rng), None, num_steps)

        # ---- intrinsic reward (ICM) ----
        ext_reward = traj.reward
        icm_metrics = {}
        if icm_net is not None:
            next_obs = jnp.concatenate([traj.obs[1:], last_obs[None]], 0)
            flat = lambda x: x.reshape((-1,) + x.shape[2:])

            def icm_loss(p):
                inv_logits, pred, phi_next = icm_net.apply(p, flat(traj.obs), flat(next_obs), flat(traj.action))
                valid = 1.0 - flat(traj.done).astype(jnp.float32)
                fwd_err = 0.5 * jnp.sum((pred - jax.lax.stop_gradient(phi_next)) ** 2, -1)
                inv_loss = optax.softmax_cross_entropy_with_integer_labels(inv_logits, flat(traj.action))
                denom = jnp.maximum(valid.sum(), 1.0)
                loss = (config["icm_forward_coef"] * (fwd_err * valid).sum() / denom
                        + config["icm_inverse_coef"] * (inv_loss * valid).sum() / denom)
                return loss, (fwd_err * valid, (fwd_err * valid).sum() / denom, (inv_loss * valid).sum() / denom)

            (_, (intrinsic, fwd_loss, inv_loss)), grads = jax.value_and_grad(icm_loss, has_aux=True)(icm_state.params)
            icm_state = icm_state.apply_gradients(grads=grads)
            raw = jax.lax.stop_gradient(intrinsic).reshape(ext_reward.shape)
            if icm_mode == "mean":
                moments = update_moments(moments, raw)
                scale = moments.mean
            elif icm_mode == "std":
                moments = update_moments(moments, raw)
                scale = jnp.sqrt(moments.var)
            elif icm_mode == "ema":
                moments = update_ema(moments, raw, config["icm_ema_decay"])
                scale = moments.mean
            else:
                scale = jnp.ones(())
            intrinsic = raw / (scale + 1e-8)
            bonus = config["icm_reward_coef"] * intrinsic
            traj = traj._replace(reward=ext_reward + bonus)
            bonus_sum, ext_sum = jnp.abs(bonus).sum(), jnp.abs(ext_reward).sum()
            icm_metrics = {"icm/forward_loss": fwd_loss, "icm/inverse_loss": inv_loss,
                           "icm/intrinsic_reward": intrinsic.mean(), "icm/extrinsic_reward": ext_reward.mean(),
                           "icm/raw_error_mean": raw.mean(), "icm/raw_error_max": raw.max(), "icm/scale": scale,
                           "icm/bonus_mean": bonus.mean(), "icm/bonus_max": bonus.max(),
                           # share of |reward| that is curiosity bonus; near 1 means the bonus has taken over
                           "icm/bonus_share": bonus_sum / jnp.maximum(bonus_sum + ext_sum, 1e-8)}

        # ---- GAE ----
        _, _, last_val, _ = apply_net(train_state.params, hstate,
                                      last_obs[None] if rnn else last_obs, last_done[None] if rnn else None)
        last_val = last_val[0] if rnn else last_val

        advantages, targets = compute_gae(traj.reward, traj.value, traj.done, last_val,
                                          config["gamma"], config["gae_lambda"])

        # ---- PPO update ----
        def loss_fn(params, h0, batch, adv, tgt):
            if rnn:
                _, pi, value, aux = apply_net(params, h0, batch.obs, batch.reset)
            else:
                _, pi, value, aux = apply_net(params, None, batch.obs, None)
            log_prob = pi.log_prob(batch.action)
            v_clipped = batch.value + jnp.clip(value - batch.value, -config["clip_eps"], config["clip_eps"])
            value_loss = 0.5 * jnp.maximum((value - tgt) ** 2, (v_clipped - tgt) ** 2).mean()
            ratio = jnp.exp(log_prob - batch.log_prob)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            actor_loss = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - config["clip_eps"],
                                                             1 + config["clip_eps"]) * adv).mean()
            entropy = pi.entropy().mean()
            total = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy \
                + config["moe_aux_coef"] * aux
            return total, (value_loss, actor_loss, entropy, aux)

        def epoch(carry, _):
            train_state, rng = carry
            rng, k = jax.random.split(rng)
            nmb = config["num_minibatches"]
            batch = traj._replace(info={})  # info is only needed for logging
            if rnn:
                # Keep sequences intact: shuffle environments and minibatch along the env axis
                perm = jax.random.permutation(k, num_envs)
                h = jnp.take(init_hstate, perm, axis=0)
                b, a, tg = jax.tree.map(lambda x: jnp.take(x, perm, axis=1), (batch, advantages, targets))

                def split_envs(x):  # (T, B, ...) -> (nmb, T, B // nmb, ...)
                    return jnp.swapaxes(x.reshape((x.shape[0], nmb, -1) + x.shape[2:]), 0, 1)

                mbs = (h.reshape((nmb, -1) + h.shape[1:]),) + tuple(jax.tree.map(split_envs, (b, a, tg)))
            else:
                batch_size = num_steps * num_envs
                perm = jax.random.permutation(k, batch_size)
                flat = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (batch, advantages, targets))
                flat = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), flat)
                b, a, tg = jax.tree.map(lambda x: x.reshape((nmb, -1) + x.shape[1:]), flat)
                mbs = (jnp.zeros((nmb,)), b, a, tg)

            def minibatch(train_state, mb):
                h0, b, a, tg = mb
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, h0, b, a, tg)
                return train_state.apply_gradients(grads=grads), (loss, *aux)

            train_state, losses = jax.lax.scan(minibatch, train_state, mbs)
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(epoch, (train_state, rng), None, config["update_epochs"])

        # ---- metrics ----
        info = traj.info
        done = info["returned_episode"].astype(jnp.float32)
        n_eps = done.sum()
        denom = jnp.maximum(n_eps, 1.0)
        metrics = {
            "episodes": n_eps,
            "episode_return": (info["returned_episode_returns"] * done).sum() / denom,
            "episode_length": (info["returned_episode_lengths"] * done).sum() / denom,
            "loss/total": losses[0].mean(), "loss/value": losses[1].mean(), "loss/actor": losses[2].mean(),
            "loss/entropy": losses[3].mean(), "loss/moe_aux": losses[4].mean(),
            "update": update_idx + 1,
            "env_steps": (update_idx + 1) * num_steps * num_envs,
        }
        rates = []
        for key, value in info.items():
            if key.startswith("Achievements/"):
                metrics[key] = value.sum() / denom  # success rate in %, info is already scaled by done * 100
                rates.append(metrics[key])
        if rates:
            rates = jnp.stack(rates)
            metrics["achievements"] = (rates / 100.0).sum()  # mean distinct achievements per finished episode
            # exploration: how many achievement types were reached at all in this update, and the score
            metrics["achievement_coverage"] = (rates > 0).sum().astype(jnp.float32)
            metrics["craftax_score"] = craftax_score(rates)
        metrics.update(icm_metrics)
        if log_fn is not None:
            jax.debug.callback(log_fn, metrics, update_idx)
        runner = (train_state, icm_state, moments, env_state, last_obs, last_done, hstate, rng)
        return runner, metrics

    def train(rng):
        runner, metrics = jax.lax.scan(update_step, init_runner(rng), jnp.arange(num_updates))
        return {"runner_state": runner, "metrics": metrics}

    def run_updates(runner, start, num):
        """Run `num` updates starting at update index `start` (for chunked / resumable training)"""
        return jax.lax.scan(update_step, runner, start + jnp.arange(num))

    train.config = config
    train.num_updates = num_updates
    train.init = init_runner
    train.run_updates = jax.jit(run_updates, static_argnums=(2,))
    return train


def save_runner(path, runner, next_update: int):
    """Save the full training state (params, optimizer, env states, RNG) to an .npz file"""
    leaves = jax.tree_util.tree_leaves(jax.device_get(runner))
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, next_update=np.asarray(next_update), **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)})
    os.replace(tmp, path)


def load_runner(path, template):
    """Restore a runner saved by `save_runner` using `template` (e.g. `train.init(rng)`) for the structure"""
    treedef = jax.tree_util.tree_structure(template)
    with np.load(path) as f:
        leaves = [jnp.asarray(f[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        next_update = int(f["next_update"])
    return jax.tree_util.tree_unflatten(treedef, leaves), next_update
