"""
Actor-critic policy: visual encoder -> (optional trajectory features) -> MoE trunk -> heads.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoders import build_encoder
from .moe import Trunk
from .time_embedding import TrajectoryEncoder


def _ortho_init(module: nn.Module, gain: float):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    """PPO actor-critic with an MoE trunk shared by the policy and value heads"""

    def __init__(self, num_actions: int, encoder: nn.Module, hidden_dim: int = 512, num_layers: int = 2,
                 trunk: str = "moe", num_experts: int = 4, top_k: int = 2,
                 trajectory_encoder: Optional[TrajectoryEncoder] = None):
        super().__init__()
        self.encoder = encoder
        self.trajectory_encoder = trajectory_encoder
        dim = encoder.feature_dim + (trajectory_encoder.out_dim if trajectory_encoder else 0)
        self.trunk = Trunk(dim, hidden_dim, num_layers=num_layers, kind=trunk,
                           num_experts=num_experts, top_k=top_k)
        self.actor = nn.Linear(dim, num_actions)
        self.critic = nn.Linear(dim, 1)
        _ortho_init(self.actor, 0.01)
        _ortho_init(self.critic, 1.0)

    def features(self, obs: torch.Tensor, trajectory: Optional[Dict[str, torch.Tensor]] = None):
        x = self.encoder(obs)
        if self.trajectory_encoder is not None:
            if trajectory is None:
                raise ValueError("this policy expects trajectory inputs (positions, timestamps)")
            x = torch.cat([x, self.trajectory_encoder(**trajectory)], dim=-1)
        return self.trunk(x)

    def forward(self, obs: torch.Tensor, trajectory=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, aux = self.features(obs, trajectory)
        return self.actor(h), self.critic(h).squeeze(-1), aux

    @torch.no_grad()
    def act(self, obs: torch.Tensor, trajectory=None, deterministic: bool = False):
        logits, value, _ = self(obs, trajectory)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, trajectory=None):
        logits, value, aux = self(obs, trajectory)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, aux


def build_actor_critic(model_cfg: Dict[str, Any], num_actions: int, image_size: int = 64) -> ActorCritic:
    """Build an ActorCritic from a plain config dict (as stored in checkpoints)"""
    enc_cfg = dict(model_cfg.get("encoder", {}))
    encoder = build_encoder(enc_cfg.pop("name", "cnn"), image_size=image_size, **enc_cfg)
    traj_cfg = model_cfg.get("trajectory")
    trajectory_encoder = TrajectoryEncoder(**traj_cfg) if traj_cfg else None
    return ActorCritic(
        num_actions,
        encoder,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_layers=model_cfg.get("num_layers", 2),
        trunk=model_cfg.get("trunk", "moe"),
        num_experts=model_cfg.get("num_experts", 4),
        top_k=model_cfg.get("top_k", 2),
        trajectory_encoder=trajectory_encoder,
    )
