"""Behavior cloning of the diffusion policy from recorded demonstrations"""

import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .rollout import ActionChunkDataset


def train_diffusion_bc(model, data: Dict[str, np.ndarray], steps: int = 10_000, batch_size: int = 128,
                       lr: float = 3e-4, device: str = 'cpu', log_every: int = 100,
                       log_fn: Optional[Callable[[Dict], None]] = None, seed: int = 0):
    dataset = ActionChunkDataset(data, model.horizon)
    if len(dataset) == 0:
        raise ValueError(f"no demonstration windows of length {model.horizon}; collect more data")
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         generator=generator)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    step, start, losses = 0, time.time(), []
    while step < steps:
        for obs, actions in loader:
            loss = model.loss(obs.to(device), actions.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % log_every == 0 or step == steps:
                metrics = {'step': step, 'loss/diffusion': float(np.mean(losses[-log_every:])),
                           'sps': step * batch_size / (time.time() - start)}
                if log_fn:
                    log_fn(metrics)
                print(f"[Diffusion BC] step {step}/{steps} loss={metrics['loss/diffusion']:.4f}")
            if step >= steps:
                break
    model.eval()
    return model
