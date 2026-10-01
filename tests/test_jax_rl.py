"""Tests for the JAX Craftax path (skipped without jax/flax/craftax)"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("craftax")

import jax.numpy as jnp  # noqa: E402

from diffu_moe_vlm.jax_rl.networks import ActorCriticRNN, MoEDense, ScannedGRU  # noqa: E402
from diffu_moe_vlm.jax_rl.ppo import RunningMoments, compute_gae, make_train, update_moments  # noqa: E402
from diffu_moe_vlm.jax_rl.wrappers import LogWrapper, OptimisticResetBatchEnv  # noqa: E402


def numpy_gae(rewards, values, dones, last_value, gamma, lam):
    adv = np.zeros_like(rewards)
    gae = np.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        nd = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nd - values[t]
        gae = delta + gamma * lam * nd * gae
        adv[t] = gae
        next_value = values[t]
    return adv


def test_gae_matches_numpy_reference():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=(7, 3)).astype(np.float32), rng.normal(size=(7, 3)).astype(np.float32)
    d = rng.random((7, 3)) < 0.3
    last = rng.normal(size=3).astype(np.float32)
    adv, tgt = compute_gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(d), jnp.asarray(last), 0.99, 0.8)
    np.testing.assert_allclose(np.asarray(adv), numpy_gae(r, v, d.astype(np.float32), last, 0.99, 0.8), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(tgt), np.asarray(adv) + v, rtol=1e-6)


def test_running_moments_match_numpy():
    rng = np.random.default_rng(1)
    chunks = [rng.exponential(size=50).astype(np.float32) for _ in range(4)]
    m = RunningMoments(jnp.zeros(()), jnp.ones(()), jnp.asarray(1e-4))
    for c in chunks:
        m = update_moments(m, jnp.asarray(c))
    allx = np.concatenate(chunks)
    assert abs(float(m.mean) - allx.mean()) < 1e-3 and abs(float(m.var) - allx.var()) < 1e-2


class CounterEnv:
    """Minimal env exposing step_env/reset_env: episode ends after 3 steps; reset draws a random id"""

    def reset_env(self, key, params):
        state = {"t": jnp.zeros((), jnp.int32), "id": jax.random.randint(key, (), 0, 1_000_000)}
        return jnp.zeros(2), state

    def reset(self, key, params):
        return self.reset_env(key, params)

    def step_env(self, key, state, action, params):
        state = {"t": state["t"] + 1, "id": state["id"]}
        done = state["t"] >= 3
        return jnp.ones(2) * state["t"], state, jnp.float32(1.0), done, {}


def test_optimistic_resets_restart_done_envs_with_distinct_worlds():
    env = LogWrapper(OptimisticResetBatchEnv(CounterEnv(), num_envs=8, reset_ratio=2))
    obs, state = env.reset(jax.random.PRNGKey(0), None)
    actions = jnp.zeros(8, jnp.int32)
    for i in range(3):
        obs, state, reward, done, info = env.step(jax.random.PRNGKey(i + 1), state, actions, None)
    assert bool(done.all())
    assert (np.asarray(state.env_state["t"]) == 0).all()                  # every env restarted
    assert len(set(np.asarray(state.env_state["id"]).tolist())) >= 4      # 4 fresh worlds for 8 envs
    assert np.allclose(np.asarray(info["returned_episode_returns"]), 3.0)
    assert np.allclose(np.asarray(info["returned_episode_lengths"]), 3)


def test_moe_dense_shapes_and_balance_loss():
    layer = MoEDense(features=16, num_experts=4, top_k=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 8))
    params = layer.init(jax.random.PRNGKey(1), x)
    y, aux = layer.apply(params, x)
    assert y.shape == (64, 16) and 1.0 <= float(aux) <= 4.0


def test_rnn_resets_hidden_state_on_episode_boundary():
    net = ActorCriticRNN(action_dim=5, width=16)
    obs = jax.random.normal(jax.random.PRNGKey(0), (3, 2, 10))
    h0 = ScannedGRU.initialize_carry(2, 16)
    params = net.init(jax.random.PRNGKey(1), h0, (obs, jnp.zeros((3, 2), bool)))
    warm = jnp.ones((2, 16))
    _, pi_reset, v_reset, _ = net.apply(params, warm, (obs[:1], jnp.ones((1, 2), bool)))
    _, pi_fresh, v_fresh, _ = net.apply(params, h0, (obs[:1], jnp.zeros((1, 2), bool)))
    np.testing.assert_allclose(np.asarray(v_reset), np.asarray(v_fresh), rtol=1e-6)
    _, _, v_warm, _ = net.apply(params, warm, (obs[:1], jnp.zeros((1, 2), bool)))
    assert not np.allclose(np.asarray(v_warm), np.asarray(v_fresh))


@pytest.mark.parametrize("rnn", [False, True])
def test_training_is_deterministic_given_seed(rnn):
    cfg = dict(env_name="Craftax-Classic-Symbolic-v1", num_envs=16, num_steps=8, num_minibatches=2,
               update_epochs=1, layer_size=32, total_timesteps=16 * 8 * 2, reset_ratio=4, rnn=rnn)
    train = jax.jit(make_train(cfg))
    a, b, c = (train(jax.random.PRNGKey(s))["metrics"] for s in (0, 0, 1))
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))
    assert any(not np.array_equal(np.asarray(a[k]), np.asarray(c[k])) for k in ("loss/total", "loss/value"))
    assert np.isfinite(np.asarray(a["loss/total"])).all()


def test_resume_from_checkpoint_is_bitwise_identical(tmp_path):
    from diffu_moe_vlm.jax_rl.ppo import load_runner, save_runner

    cfg = dict(env_name="Craftax-Classic-Symbolic-v1", num_envs=16, num_steps=8, num_minibatches=2,
               update_epochs=1, layer_size=32, total_timesteps=16 * 8 * 4, reset_ratio=4, rnn=True)
    train = make_train(cfg)
    runner = train.init(jax.random.PRNGKey(3))
    straight, m_straight = train.run_updates(runner, 0, 4)

    half, m_first = train.run_updates(runner, 0, 2)
    save_runner(tmp_path / "ckpt.npz", half, 2)
    restored, next_update = load_runner(tmp_path / "ckpt.npz", train.init(jax.random.PRNGKey(99)))
    resumed, m_second = train.run_updates(restored, next_update, 2)

    assert next_update == 2
    for key in m_straight:
        np.testing.assert_array_equal(np.asarray(m_straight[key]),
                                      np.concatenate([np.asarray(m_first[key]), np.asarray(m_second[key])]))
    for a, b in zip(jax.tree_util.tree_leaves(straight[0].params), jax.tree_util.tree_leaves(resumed[0].params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
