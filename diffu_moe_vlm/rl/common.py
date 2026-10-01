"""Shared helpers for the training / evaluation entry points"""

import functools
import json
import random
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ..crafter_env import CrafterEnv
from .vec_env import VecEnv


def resolve_device(name: str = 'auto') -> str:
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def seed_everything(seed: int, torch_threads: Optional[int] = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_threads:
        # Leave CPU cores for environment worker processes
        torch.set_num_threads(torch_threads)


def make_crafter_envs(num_envs: int, seed: int = 0, size: int = 64, length: int = 10000,
                      num_workers: Optional[int] = None) -> VecEnv:
    env_fn = functools.partial(CrafterEnv, size=size, length=length)
    return VecEnv([env_fn] * num_envs, seeds=[seed + i for i in range(num_envs)], num_workers=num_workers)


def make_logger(wandb_cfg: Optional[Dict], run_config: Dict, jsonl_path=None) -> Callable[[Dict], None]:
    """Log metrics to a JSONL file and, if enabled, to Weights & Biases"""
    wandb_run = None
    if wandb_cfg and wandb_cfg.get('enabled', False):
        import wandb
        wandb_run = wandb.init(project=wandb_cfg.get('project', 'diffu-moe-vlm-crafter'),
                               name=wandb_cfg.get('name'), tags=list(wandb_cfg.get('tags', [])),
                               config=run_config)
    jsonl = None
    if jsonl_path is not None:
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        jsonl = open(jsonl_path, 'a')

    def log(metrics: Dict):
        if jsonl:
            jsonl.write(json.dumps(metrics) + "\n")
            jsonl.flush()
        if wandb_run:
            wandb_run.log(metrics, step=int(metrics.get('step', 0)))

    return log


def write_json(path, data: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
