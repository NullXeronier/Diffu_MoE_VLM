"""
Diffusion policy over discrete action sequences (DDPM).

An action chunk of horizon H is represented as one-hot vectors scaled to
[-1, 1], shape (H, A). A conditional denoiser sees the noisy chunk, the
diffusion step and the observation embedding and predicts either the clean
chunk (`prediction_type="sample"`, default: much more reliable for one-hot
targets, where epsilon-prediction barely constrains x0 at high noise levels)
or the noise (`"epsilon"`). Sampling runs the reverse process (optionally
strided, DDIM-style) and decodes with argmax.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder
from .time_embedding import SinusoidalTimeEmbedding


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(1e-5, 0.999)


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, dim * 2)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return x + self.mlp(self.norm(x) * (1 + scale) + shift)


class Denoiser(nn.Module):
    def __init__(self, horizon: int, num_actions: int, cond_dim: int, hidden_dim: int = 512, depth: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(horizon * num_actions, hidden_dim)
        self.time = nn.Sequential(SinusoidalTimeEmbedding(128), nn.Linear(128, cond_dim), nn.SiLU())
        self.blocks = nn.ModuleList([FiLMResBlock(hidden_dim, cond_dim) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, horizon * num_actions))

    def forward(self, x_t, t, obs_emb):
        cond = obs_emb + self.time(t)
        h = self.in_proj(x_t.flatten(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).view_as(x_t)


class DiffusionPolicy(nn.Module):
    """Observation-conditioned DDPM over (H, A) one-hot action chunks"""

    def __init__(self, num_actions: int, encoder: nn.Module, horizon: int = 8, num_diffusion_steps: int = 50,
                 hidden_dim: int = 512, depth: int = 4, prediction_type: str = "sample"):
        super().__init__()
        if prediction_type not in ("sample", "epsilon"):
            raise ValueError(f"prediction_type must be 'sample' or 'epsilon', got {prediction_type!r}")
        self.prediction_type = prediction_type
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.encoder = encoder
        self.denoiser = Denoiser(horizon, num_actions, encoder.feature_dim, hidden_dim, depth)
        betas = cosine_beta_schedule(num_diffusion_steps)
        alphas_bar = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions, self.num_actions).float() * 2 - 1

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs: (B, H, W, 3) uint8; actions: (B, horizon) long"""
        x0 = self.encode_actions(actions)
        t = torch.randint(0, self.num_diffusion_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alphas_bar[t].view(-1, 1, 1)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.denoiser(x_t, t.float(), self.encoder(obs))
        return F.mse_loss(pred, x0 if self.prediction_type == "sample" else noise)

    def _predict(self, x, t: int, obs_emb):
        """Return (x0_hat clamped to [-1, 1], eps_hat) at integer step t"""
        tt = torch.full((x.shape[0],), float(t), device=x.device)
        out = self.denoiser(x, tt, obs_emb)
        ab = self.alphas_bar[t]
        if self.prediction_type == "sample":
            x0 = out.clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        else:
            eps = out
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        return x0, eps

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, num_steps: Optional[int] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Return (B, horizon) actions. `num_steps` < T uses a strided deterministic (DDIM) schedule."""
        obs_emb = self.encoder(obs)
        shape = (obs.shape[0], self.horizon, self.num_actions)
        x = torch.randn(shape, device=obs.device, generator=generator)
        T = self.num_diffusion_steps
        if num_steps is None or num_steps >= T:
            for t in reversed(range(T)):
                x0, _ = self._predict(x, t, obs_emb)
                if t == 0:
                    x = x0
                    break
                # Posterior q(x_{t-1} | x_t, x0)
                ab, ab_prev, beta = self.alphas_bar[t], self.alphas_bar[t - 1], self.betas[t]
                mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0 + ((1 - beta).sqrt() * (1 - ab_prev) / (1 - ab)) * x
                var = beta * (1 - ab_prev) / (1 - ab)
                x = mean + var.sqrt() * torch.randn(shape, device=obs.device, generator=generator)
        else:
            steps = torch.linspace(T - 1, 0, num_steps).round().long().tolist()
            for i, t in enumerate(steps):
                x0, eps = self._predict(x, t, obs_emb)
                if i + 1 == len(steps):
                    x = x0
                    break
                ab_prev = self.alphas_bar[steps[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.argmax(dim=-1)


def build_diffusion_policy(model_cfg, num_actions: int, image_size: int = 64) -> DiffusionPolicy:
    enc_cfg = dict(model_cfg.get("encoder", {}))
    encoder = build_encoder(enc_cfg.pop("name", "cnn"), image_size=image_size, **enc_cfg)
    return DiffusionPolicy(
        num_actions, encoder,
        horizon=model_cfg.get("horizon", 8),
        num_diffusion_steps=model_cfg.get("num_diffusion_steps", 50),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        depth=model_cfg.get("depth", 4),
        prediction_type=model_cfg.get("prediction_type", "sample"),
    )
