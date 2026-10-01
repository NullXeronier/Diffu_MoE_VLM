# Craftax reproducibility test

Goal: reproduce the reference W&B runs (`Craftax-Symbolic-v1-1000M`, `Craftax-Symbolic-v1-PPO_RNN-1000M`,
see `docs/craftax_reference.json`) with `train_craftax.py`.

## What was tested here (CPU, 4 cores, no GPU)

| Check | Result |
|---|---|
| Unit: GAE vs. NumPy reference, running moments, optimistic resets (fresh distinct worlds for done envs), MoE balance loss, GRU state reset at episode boundaries | pass (`tests/test_jax_rl.py`) |
| Determinism: same seed gives bit-identical metrics (PPO and PPO-RNN); different seeds differ | pass |
| Resume: 2 + save/load + 2 updates is bit-identical to 4 straight updates (metrics and params) | pass |
| Real-scale determinism: baseline config (1024 envs x 64 steps), one uninterrupted `scan` vs. 5-update chunks with checkpoints, resumed across processes: 390 logged values at updates 5-30 | 0 differences |
| Learning, PPO, 100 updates = 6.55M steps (0.66% of the reference budget) | return 1.49 -> 6.66, achievements/episode 2.4 -> 7.6 |
| Learning, PPO-RNN, 50 updates = 3.28M steps | return 1.49 -> 5.15 (PPO at the same step: 5.47) |

Throughput on this CPU: ~1,150-1,250 env steps/s (reference GPU runs: 50k-95k).

### Early learning (single seed)

| Updates (env steps) | Model | Return | Ach./ep. | collect_wood | place_table | make_wood_pickaxe | eat_cow | collect_stone |
|---|---|---|---|---|---|---|---|---|
| 10 (0.66M) | PPO | 2.45 | 3.35 | 65.0 | 32.9 | 0.8 | 0.8 | 0.4 |
| 10 (0.66M) | PPO-RNN | 2.57 | 3.47 | 62.6 | 32.6 | 1.1 | 1.1 | 0.0 |
| 50 (3.28M) | PPO | 5.47 | 6.37 | 96.4 | 92.3 | 39.4 | 28.1 | 2.3 |
| 50 (3.28M) | PPO-RNN | 5.15 | 6.05 | 95.8 | 84.1 | 19.3 | 18.4 | 4.2 |
| 100 (6.55M) | PPO | 6.66 | 7.56 | 97.1 | 89.5 | 50.0 | 70.5 | 17.6 |

This matches the start of the reference curves: the basic achievements (collect_wood, place_table,
collect_sapling, place_plant) saturate first, wake_up starts near 100% and declines later, and the
wood/stone tool chain follows. The reference PPO-RNN advantage appears over hundreds of millions of
steps; at 3M steps the two are within single-seed noise. Raw logs: `docs/craftax_runs/*.jsonl`.

## Not tested here

The end-of-training numbers (PPO return ~26-28, PPO-RNN ~37) need the full 1e9 steps, about 3-4 hours
per run on one GPU (the reference runs), versus ~10 days on this CPU. To finish the test on a GPU:

```bash
pip install -e ".[jax]" && pip install -U "jax[cuda12]"
python train_craftax.py output_dir=runs/ppo seed=0
python train_craftax.py algo.rnn=true output_dir=runs/ppo_rnn seed=0
python compare_craftax_run.py runs/ppo/metrics.jsonl --ref ppo
python compare_craftax_run.py runs/ppo_rnn/metrics.jsonl --ref ppo_rnn
```

A run is reproduced when `compare_craftax_run.py` reports the return, length, achievements/episode
and the per-achievement rates inside the reference ranges (they are read off chart images and are
approximate; use 2-3 seeds). If a run is interrupted, re-run the same command to resume.

Hyper-parameters follow the Craftax PPO baselines (1024 envs, 64 steps, lr 2e-4, gamma 0.99,
lambda 0.8, 3x512 tanh MLP / GRU-512); the reference runs' own config was not available, so confirm
these against it if possible.
