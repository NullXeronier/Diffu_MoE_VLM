"""
Proximal Policy Optimization for the MoE actor-critic.

Standard clipped PPO with GAE. The MoE load-balancing loss is added to the
objective with `moe_aux_coef`. Episode returns and Crafter achievements are
tracked from the vectorized environment and reported per update.
Time-limit truncation is treated like termination when bootstrapping, which
is negligible for Crafter's 10k-step episodes.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..crafter_env import AchievementTracker, format_achievements
from .vec_env import VecEnv


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    num_envs: int = 8
    rollout_length: int = 128
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.65
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    moe_aux_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    log_window: int = 100  # episodes used for logged success rates


class PPOTrainer:
    def __init__(self, model: nn.Module, envs: VecEnv, cfg: PPOConfig, device: str = 'cpu',
                 log_fn: Optional[Callable[[Dict], None]] = None,
                 checkpoint_fn: Optional[Callable[[int], None]] = None, checkpoint_every: int = 0):
        self.model = model.to(device)
        self.envs = envs
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self.tracker = AchievementTracker()
        self.log_fn = log_fn or (lambda metrics: None)
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = checkpoint_every
        self.global_step = 0

    def _collect(self, obs: np.ndarray):
        cfg, n = self.cfg, self.envs.num_envs
        buf_obs = np.zeros((cfg.rollout_length, n) + obs.shape[1:], dtype=np.uint8)
        buf_actions = torch.zeros(cfg.rollout_length, n, dtype=torch.long)
        buf_logp = torch.zeros(cfg.rollout_length, n)
        buf_values = torch.zeros(cfg.rollout_length, n)
        buf_rewards = torch.zeros(cfg.rollout_length, n)
        buf_dones = torch.zeros(cfg.rollout_length, n)

        self.model.eval()
        for t in range(cfg.rollout_length):
            buf_obs[t] = obs
            action, logp, value = self.model.act(torch.as_tensor(obs, device=self.device))
            obs, reward, done, episodes = self.envs.step(action.cpu().numpy())
            buf_actions[t], buf_logp[t], buf_values[t] = action.cpu(), logp.cpu(), value.cpu()
            buf_rewards[t] = torch.as_tensor(reward)
            buf_dones[t] = torch.as_tensor(done)
            for ep in episodes:
                if ep is not None:
                    self.tracker.add_episode(ep['achievements'], ep['return'], ep['length'])
        self.global_step += cfg.rollout_length * n

        with torch.no_grad():
            _, last_value, _ = self.model(torch.as_tensor(obs, device=self.device))
        advantages = torch.zeros_like(buf_rewards)
        gae = torch.zeros(n)
        next_value = last_value.cpu()
        for t in reversed(range(cfg.rollout_length)):
            not_done = 1.0 - buf_dones[t]
            delta = buf_rewards[t] + cfg.gamma * next_value * not_done - buf_values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            advantages[t] = gae
            next_value = buf_values[t]
        returns = advantages + buf_values
        batch = {
            'obs': buf_obs.reshape((-1,) + buf_obs.shape[2:]),
            'actions': buf_actions.flatten(),
            'logp': buf_logp.flatten(),
            'advantages': advantages.flatten(),
            'returns': returns.flatten(),
        }
        return obs, batch

    def _update(self, batch) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()
        size = batch['actions'].shape[0]
        mb_size = size // cfg.minibatches
        stats = {'loss/policy': [], 'loss/value': [], 'loss/entropy': [], 'loss/moe_aux': [], 'ppo/clip_frac': [],
                 'ppo/approx_kl': []}
        for _ in range(cfg.epochs):
            perm = torch.randperm(size)
            for start in range(0, size - mb_size + 1, mb_size):
                idx = perm[start:start + mb_size]
                obs = torch.as_tensor(batch['obs'][idx.numpy()], device=self.device)
                actions = batch['actions'][idx].to(self.device)
                old_logp = batch['logp'][idx].to(self.device)
                adv = batch['advantages'][idx].to(self.device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ret = batch['returns'][idx].to(self.device)

                logp, entropy, value, aux = self.model.evaluate_actions(obs, actions)
                ratio = (logp - old_logp).exp()
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                value_loss = 0.5 * (value - ret).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = (policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
                        + cfg.moe_aux_coef * aux)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats['loss/policy'].append(policy_loss.item())
                    stats['loss/value'].append(value_loss.item())
                    stats['loss/entropy'].append(entropy_mean.item())
                    stats['loss/moe_aux'].append(float(aux))
                    stats['ppo/clip_frac'].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    stats['ppo/approx_kl'].append((old_logp - logp).mean().item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train(self) -> AchievementTracker:
        cfg = self.cfg
        num_updates = max(1, cfg.total_steps // (cfg.rollout_length * self.envs.num_envs))
        obs = self.envs.reset()
        start = time.time()
        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                for group in self.optimizer.param_groups:
                    group['lr'] = cfg.lr * (1.0 - (update - 1) / num_updates)
            obs, batch = self._collect(obs)
            stats = self._update(batch)

            summary = self.tracker.summary(last=cfg.log_window)
            metrics = {'step': self.global_step, 'sps': self.global_step / (time.time() - start), **stats, **summary}
            if hasattr(self.model, 'trunk'):
                for layer, load in enumerate(self.model.trunk.expert_load()):
                    for e, frac in enumerate(load):
                        metrics[f'moe/layer{layer}_expert{e}_load'] = frac
            self.log_fn(metrics)
            print(f"[PPO] update {update}/{num_updates} step={self.global_step} sps={metrics['sps']:.0f} "
                  f"episodes={summary['episodes']} return={summary['return_mean']:.2f} "
                  f"score={summary['score']:.2f} entropy={stats['loss/entropy']:.3f}")
            if self.checkpoint_fn and self.checkpoint_every and update % self.checkpoint_every == 0:
                self.checkpoint_fn(self.global_step)

        if self.tracker.episodes:
            print("[PPO] achievements (last window): " +
                  format_achievements(self.tracker.success_rates(last=cfg.log_window)))
        return self.tracker
