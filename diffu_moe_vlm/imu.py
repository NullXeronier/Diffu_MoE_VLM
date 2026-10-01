"""
3D trajectory / IMU data utilities.

Trajectory files are .npz archives with:
    positions  (N, T, J, 3) float  joint positions (e.g. left hand, right hand, head)
    timestamps (N, T) float        sample times in seconds (may be irregular)
    mask       (N, T) bool         optional, True for valid samples
    labels     (N,) int            optional, e.g. activity class
`make_synthetic_trajectories` produces data in this format for testing.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

JOINTS = ('left_hand', 'right_hand', 'head')
MOTIONS = ('idle', 'wave', 'chop', 'reach')


def make_synthetic_trajectories(num: int = 256, length: int = 64, rate_hz: float = 60.0,
                                jitter: float = 0.3, seed: int = 0) -> Dict[str, np.ndarray]:
    """Synthetic hand/head motions (idle, wave, chop, reach) with irregular sampling"""
    rng = np.random.default_rng(seed)
    base = np.array([[-0.25, 1.1, 0.3], [0.25, 1.1, 0.3], [0.0, 1.65, 0.0]])  # left hand, right hand, head
    positions = np.zeros((num, length, len(JOINTS), 3), dtype=np.float32)
    timestamps = np.zeros((num, length), dtype=np.float32)
    labels = rng.integers(len(MOTIONS), size=num)
    for n in range(num):
        dt = (1.0 / rate_hz) * (1 + jitter * rng.uniform(-1, 1, size=length))
        t = np.cumsum(dt) - dt[0]
        freq = rng.uniform(1.0, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        osc = np.sin(2 * np.pi * freq * t + phase)
        pos = np.repeat(base[None], length, axis=0).copy()
        motion = MOTIONS[labels[n]]
        if motion == 'wave':        # right hand oscillates sideways, raised
            pos[:, 1, 0] += 0.15 * osc
            pos[:, 1, 1] += 0.4
        elif motion == 'chop':      # both hands move up and down together
            pos[:, :2, 1] += 0.25 * osc[:, None]
        elif motion == 'reach':     # right hand extends forward and back
            pos[:, 1, 2] += 0.3 * (osc + 1) / 2
        pos += rng.normal(0, 0.01, size=pos.shape)                      # sensor noise
        pos[:, 2] += rng.normal(0, 0.005, size=(length, 3)).cumsum(0)   # head drift
        positions[n], timestamps[n] = pos, t
    return {'positions': positions, 'timestamps': timestamps,
            'mask': np.ones((num, length), dtype=bool), 'labels': labels.astype(np.int64)}


def save_trajectories(path, data: Dict[str, np.ndarray]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_trajectories(path) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    if 'positions' not in data or 'timestamps' not in data:
        raise ValueError("trajectory file needs 'positions' (N,T,J,3) and 'timestamps' (N,T)")
    data.setdefault('mask', np.ones(data['timestamps'].shape, dtype=bool))
    return data


def to_torch_batch(data: Dict[str, np.ndarray], index: Optional[np.ndarray] = None, device: str = 'cpu'):
    """Slice a trajectory dict into TrajectoryEncoder keyword arguments"""
    import torch

    index = slice(None) if index is None else index
    return {
        'positions': torch.as_tensor(data['positions'][index], dtype=torch.float32, device=device),
        'timestamps': torch.as_tensor(data['timestamps'][index], dtype=torch.float32, device=device),
        'mask': torch.as_tensor(data['mask'][index], dtype=torch.bool, device=device),
    }
