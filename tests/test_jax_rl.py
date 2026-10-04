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


# ---- ICM normalization ablation ----

def test_ema_initializes_with_first_batch_then_decays():
    from diffu_moe_vlm.jax_rl.ppo import update_ema

    m = RunningMoments(jnp.zeros(()), jnp.ones(()), jnp.asarray(1e-4))
    m = update_ema(m, jnp.full((4,), 10.0), 0.9)
    assert float(m.mean) == pytest.approx(10.0) and float(m.count) == 1
    m = update_ema(m, jnp.full((4,), 0.0), 0.9)
    assert float(m.mean) == pytest.approx(9.0) and float(m.count) == 2


def test_icm_normalizer_modes():
    from diffu_moe_vlm.jax_rl.ppo import icm_normalizer

    assert icm_normalizer({"icm_normalize": True}) == "mean"
    assert icm_normalizer({"icm_normalize": False}) == "none"
    assert icm_normalizer({"icm_normalize": "ema"}) == "ema"
    with pytest.raises(ValueError):
        icm_normalizer({"icm_normalize": "max"})


def test_craftax_score_matches_formula():
    from diffu_moe_vlm.jax_rl.ppo import craftax_score

    rates = np.array([0.0, 10.0, 100.0], np.float32)
    assert float(craftax_score(jnp.asarray(rates))) == pytest.approx(np.exp(np.log1p(rates).mean()) - 1, rel=1e-5)


@pytest.mark.parametrize("mode", ["none", "std", "mean", "ema"])
def test_icm_modes_log_bonus_and_exploration_metrics(mode):
    coef = 0.01
    cfg = dict(env_name="Craftax-Classic-Symbolic-v1", num_envs=16, num_steps=8, num_minibatches=2,
               update_epochs=1, layer_size=32, total_timesteps=16 * 8 * 2, reset_ratio=4,
               icm=True, icm_normalize=mode, icm_reward_coef=coef)
    m = jax.jit(make_train(cfg))(jax.random.PRNGKey(0))["metrics"]
    share = np.asarray(m["icm/bonus_share"])
    assert ((share >= 0) & (share <= 1)).all()
    for key in ("craftax_score", "achievement_coverage", "icm/raw_error_mean", "icm/bonus_max", "icm/scale"):
        assert np.isfinite(np.asarray(m[key])).all()
    first_bonus = float(np.asarray(m["icm/bonus_mean"])[0])
    first_raw = float(np.asarray(m["icm/raw_error_mean"])[0])
    if mode in ("mean", "ema"):     # the first batch defines the scale, so the bonus averages the coefficient
        assert first_bonus == pytest.approx(coef, rel=1e-3)
    elif mode == "none":
        assert first_bonus == pytest.approx(coef * first_raw, rel=1e-4)


def _rows(shares, returns, covered):
    rows = []
    for i, (s, r) in enumerate(zip(shares, returns)):
        row = {"update": i + 1, "env_steps": (i + 1) * 2e7, "episodes": 4, "episode_return": r,
               "craftax_score": r / 10, "achievements": r, "achievement_coverage": float(covered),
               "Achievements/a": 50.0, "Achievements/b": 10.0 if covered > 1 else 0.0}
        if s is not None:
            row.update({"icm/bonus_share": s, "icm/raw_error_mean": 1.0 + i, "icm/bonus_mean": s, "icm/bonus_max": s})
        rows.append(row)
    return rows


def test_analysis_detects_runaway_and_paired_exploration(tmp_path):
    import json as _json

    from analyze_icm_ablation import analyze, write_report

    # icm_mean: early bonus share is high (sparse extrinsic reward) but falls; it must not count as runaway
    arms = {"ppo": ([None] * 5, [1, 2, 3, 3, 3], 1), "icm_mean": ([0.8, 0.6, 0.3, 0.2, 0.1], [1, 2, 3, 4, 4], 2),
            "icm_none": ([0.2, 0.6, 0.9, 0.95, 0.97], [1, 1, 0, 0, 0], 1)}
    for arm, (shares, rets, cov) in arms.items():
        for seed in range(3):
            d = tmp_path / arm / f"seed{seed}"
            d.mkdir(parents=True)
            (d / "metrics.jsonl").write_text("\n".join(_json.dumps(r) for r in _rows(shares, rets, cov)))
    result = analyze(tmp_path)
    none, mean = result["arms"]["icm_none"]["seeds"][0], result["arms"]["icm_mean"]["seeds"][0]
    assert none["runaway_fraction"] == pytest.approx(0.8) and none["runaway"] == 1.0
    assert none["bonus_growth"] == pytest.approx(0.97 / 0.2)
    assert mean["runaway"] == 0.0 and mean["bonus_growth"] < 1.0
    assert result["comparisons"]["icm_mean"]["coverage_ever"]["positive"] == 3
    assert result["verdict"]["H1"][0] == "supported" and result["verdict"]["H2"][0] == "supported"
    assert "Verdict" in write_report(result, tmp_path) and (tmp_path / "summary.json").exists()
    assert list(result["arms"]) == ["ppo", "icm_none", "icm_mean"]


def test_analysis_refuses_verdict_on_short_runs(tmp_path):
    import json as _json

    from analyze_icm_ablation import analyze

    for arm in ("ppo", "icm_mean"):
        d = tmp_path / arm / "seed0"
        d.mkdir(parents=True)
        rows = _rows([None if arm == "ppo" else 0.1] * 3, [1, 2, 3], 1)
        for r in rows:
            r["env_steps"] = 1000.0
        (d / "metrics.jsonl").write_text("\n".join(_json.dumps(r) for r in rows))
    assert analyze(tmp_path)["verdict"]["H1"][0] == "too short"


def test_ablation_runner_builds_one_change_per_arm(tmp_path):
    from run_icm_ablation import build_runs

    runs = build_runs("gpu", ["ppo", "icm_mean", "icm_std"], [0, 1], 1e9, tmp_path)
    assert len(runs) == 6
    by_arm = {r["arm"]: r["cmd"] for r in runs if r["seed"] == 0}
    assert "algo.icm=false" in by_arm["ppo"] and "algo.icm_normalize=mean" in by_arm["icm_mean"]
    common = lambda cmd: sorted(a for a in cmd[2:] if not a.startswith(("algo.icm", "output_dir", "name")))
    assert common(by_arm["ppo"]) == common(by_arm["icm_mean"]) == common(by_arm["icm_std"])
