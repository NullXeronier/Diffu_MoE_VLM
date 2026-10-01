"""Checkpoint save/load. A checkpoint stores the model config so it can be rebuilt without Hydra."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path, model: torch.nn.Module, kind: str, model_cfg: Dict[str, Any],
                    optimizer: Optional[torch.optim.Optimizer] = None, step: int = 0,
                    extra: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'kind': kind,
        'model_cfg': model_cfg,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'step': step,
        'extra': extra or {},
    }, path)
    return path


def load_checkpoint(path, device: str = 'cpu') -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def load_policy(path, num_actions: int, device: str = 'cpu', image_size: int = 64):
    """Rebuild an ActorCritic ('ppo') or DiffusionPolicy ('diffusion') from a checkpoint"""
    from ..nn.diffusion import build_diffusion_policy
    from ..nn.policy import build_actor_critic

    ckpt = load_checkpoint(path, device)
    builder = {'ppo': build_actor_critic, 'diffusion': build_diffusion_policy}[ckpt['kind']]
    model = builder(ckpt['model_cfg'], num_actions, image_size=image_size).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt
