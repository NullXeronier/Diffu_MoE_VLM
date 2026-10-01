"""
Time embeddings and the 3D trajectory / IMU encoder.

`SinusoidalTimeEmbedding` embeds continuous times (diffusion steps or
timestamps in seconds). `TrajectoryEncoder` encodes irregularly sampled
multi-joint 3D trajectories (e.g. left hand, right hand, head from motion
capture or IMU-derived poses) into a single feature vector.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding of continuous scalar times"""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TrajectoryEncoder(nn.Module):
    """
    Encode (B, T, J, 3) joint positions sampled at (B, T) timestamps.

    Per step, positions and finite-difference velocities are projected and a
    continuous time embedding of the timestamp (relative to the first sample,
    scaled by `time_scale`) is added, so irregular sampling rates are handled.
    A transformer encoder mixes the sequence and a masked mean pool returns
    (B, out_dim).
    """

    def __init__(self, num_joints: int = 3, dim: int = 128, out_dim: int = 128, depth: int = 2,
                 heads: int = 4, time_scale: float = 100.0):
        super().__init__()
        self.num_joints = num_joints
        self.time_scale = time_scale
        self.input_proj = nn.Linear(num_joints * 6, dim)
        self.time_embed = nn.Sequential(SinusoidalTimeEmbedding(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        self.out_dim = out_dim

    def forward(self, positions: torch.Tensor, timestamps: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            positions: (B, T, J, 3) joint positions
            timestamps: (B, T) sample times in seconds (increasing)
            mask: (B, T) bool, True for valid samples (default all valid)
        """
        b, t, j, _ = positions.shape
        assert j == self.num_joints, f"expected {self.num_joints} joints, got {j}"
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=positions.device)

        rel_t = timestamps - timestamps[:, :1]
        dt = torch.diff(rel_t, dim=1, prepend=rel_t[:, :1]).clamp_min(1e-3).unsqueeze(-1).unsqueeze(-1)
        velocity = torch.diff(positions, dim=1, prepend=positions[:, :1]) / dt
        features = torch.cat([positions, velocity], dim=-1).reshape(b, t, j * 6)

        x = self.input_proj(features) + self.time_embed(rel_t * self.time_scale)
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        weights = mask.float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.out(pooled)
