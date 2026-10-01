import functools

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from diffu_moe_vlm.crafter_env import CRAFTER_ACHIEVEMENTS, AchievementTracker, crafter_score  # noqa: E402
from diffu_moe_vlm.nn.policy import build_actor_critic  # noqa: E402
from diffu_moe_vlm.rl.checkpoint import load_policy, save_checkpoint  # noqa: E402
from diffu_moe_vlm.rl.ppo import PPOConfig, PPOTrainer  # noqa: E402
from diffu_moe_vlm.rl.rollout import (ActionChunkDataset, ActorCriticRunner, RandomRunner,  # noqa: E402
                                      collect_demonstrations, evaluate)
from diffu_moe_vlm.rl.vec_env import VecEnv  # noqa: E402


class ColorBanditEnv(gym.Env):
    """One-step task: the image is dark or bright and the correct action is 0 or 1 accordingly"""

    observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
    action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target = int(self.np_random.integers(2))
        return np.full((64, 64, 3), 255 * self.target, np.uint8), {}

    def step(self, action):
        reward = 1.0 if int(action) == self.target else 0.0
        obs, _ = self.reset()
        return obs, reward, True, False, {'achievements': {'collect_wood': int(reward)}}


def small_model(num_actions):
    return build_actor_critic({"encoder": {"name": "cnn", "feature_dim": 64}, "hidden_dim": 64,
                               "num_experts": 2, "top_k": 1}, num_actions)


def test_crafter_score_and_tracker():
    assert crafter_score({name: 0.0 for name in CRAFTER_ACHIEVEMENTS}) == pytest.approx(0.0)
    assert crafter_score({name: 100.0 for name in CRAFTER_ACHIEVEMENTS}) == pytest.approx(100.0)
    tracker = AchievementTracker()
    tracker.add_episode({'eat_plant': 1, 'collect_wood': 3}, 2.0, 100)
    tracker.add_episode({'collect_wood': 1}, 1.0, 50)
    rates = tracker.success_rates()
    assert rates['collect_wood'] == 100.0 and rates['eat_plant'] == 50.0 and rates['collect_diamond'] == 0.0
    summary = tracker.summary()
    assert summary['return_mean'] == 1.5 and summary['achievements/eat_plant'] == 50.0


def test_ppo_solves_color_bandit():
    torch.manual_seed(0)
    envs = VecEnv([ColorBanditEnv] * 8, seeds=list(range(8)), num_workers=0)
    cfg = PPOConfig(total_steps=8 * 32 * 12, num_envs=8, rollout_length=32, epochs=4, minibatches=4,
                    lr=1e-3, entropy_coef=0.0, log_window=200)
    tracker = PPOTrainer(small_model(3), envs, cfg).train()
    assert np.mean(tracker.returns[-200:]) > 0.9


def test_checkpoint_roundtrip(tmp_path):
    model = small_model(17).eval()
    cfg = {"encoder": {"name": "cnn", "feature_dim": 64}, "hidden_dim": 64, "num_experts": 2, "top_k": 1}
    path = save_checkpoint(tmp_path / "ckpt.pt", model, "ppo", cfg, step=7)
    loaded, ckpt = load_policy(path, 17)
    obs = torch.randint(0, 256, (3, 64, 64, 3), dtype=torch.uint8)
    assert ckpt["step"] == 7
    assert torch.allclose(model(obs)[0], loaded(obs)[0])


def test_action_chunk_windows_stay_inside_episodes():
    data = {'obs': np.zeros((7, 64, 64, 3), np.uint8), 'actions': np.arange(7),
            'episode': np.array([0, 0, 0, 1, 1, 1, 1])}
    ds = ActionChunkDataset(data, horizon=3)
    assert [ds[i][1].tolist() for i in range(len(ds))] == [[0, 1, 2], [3, 4, 5], [4, 5, 6]]


def test_crafter_wrapper_evaluation_and_demos():
    pytest.importorskip("crafter")
    from diffu_moe_vlm.crafter_env import CrafterEnv

    env = CrafterEnv(length=5)
    obs, info = env.reset(seed=0)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and set(info['achievements']) == set(CRAFTER_ACHIEVEMENTS)

    envs = VecEnv([functools.partial(CrafterEnv, length=20)] * 2, seeds=[0, 1], num_workers=0)
    tracker = evaluate(RandomRunner(17), envs, num_episodes=3)
    assert len(tracker.episodes) == 3 and all(length <= 20 for length in tracker.lengths)

    data = collect_demonstrations(ActorCriticRunner(small_model(17).eval()), envs, num_steps=40)
    assert data['obs'].shape[0] == data['actions'].shape[0] == data['episode'].shape[0] >= 40
