"""
Mixture-of-Experts layers with top-k gating and a load-balancing loss.

Experts are evaluated densely and combined with the sparse top-k gate weights,
which is simple and efficient for the small expert counts used here.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Top-k gated mixture of MLP experts on (B, D) inputs"""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2, noisy_gating: bool = True):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.noise = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            for _ in range(num_experts)
        ])
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.noise.weight)
        self.register_buffer("last_load", torch.zeros(num_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise(x))
        probs = logits.softmax(dim=-1)                                  # (B, E)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)              # (B, k)
        weights = top_vals / top_vals.sum(dim=-1, keepdim=True)

        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, D)
        chosen = expert_out.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        y = (weights.unsqueeze(-1) * chosen).sum(dim=1)

        # Switch-Transformer load-balancing loss: E * sum_i f_i * P_i (1.0 when perfectly balanced)
        dispatch = F.one_hot(top_idx[:, 0], self.num_experts).float()
        load = dispatch.mean(dim=0)
        importance = probs.mean(dim=0)
        aux_loss = self.num_experts * (load * importance).sum()
        self.last_load = load.detach()
        return y, aux_loss


class MoEBlock(nn.Module):
    """Pre-norm residual block around an MoE layer"""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.moe = MoELayer(dim, hidden_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, aux = self.moe(self.norm(x))
        return x + y, aux


class MLPBlock(nn.Module):
    """Dense counterpart of MoEBlock (for ablations); aux loss is zero"""

    def __init__(self, dim: int, hidden_dim: int, **_):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + self.mlp(self.norm(x)), x.new_zeros(())


class Trunk(nn.Module):
    """Stack of MoE (or dense) residual blocks returning features and summed aux loss"""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 2, kind: str = "moe",
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        if kind not in ("moe", "mlp"):
            raise ValueError(f"Unknown trunk {kind!r} (expected moe or mlp)")
        block = MoEBlock if kind == "moe" else MLPBlock
        self.blocks = nn.ModuleList([
            block(dim, hidden_dim, num_experts=num_experts, top_k=top_k) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x)
            aux = aux + block_aux
        return self.norm(x), aux

    def expert_load(self):
        return [b.moe.last_load.tolist() for b in self.blocks if isinstance(b, MoEBlock)]
