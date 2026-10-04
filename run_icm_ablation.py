"""
Does moving-mean normalization of the ICM bonus stop the intrinsic reward from
running away, and does it actually improve exploration?

Runs the same Craftax PPO with one change per arm (train_craftax.py underneath),
then compares the arms with analyze_icm_ablation.py:

    ppo        no curiosity bonus (baseline)
    icm_none   raw forward-model error           (expected: bonus runs away)
    icm_std    error / running std               (expected: still too large)
    icm_mean   error / running mean   (default)  (claim: bounded, explores more)
    icm_ema    error / exponential moving mean   (follows a shrinking error)

All ICM arms use the same icm_reward_coef, so only the normalization differs.

GPU (the experiment; ~3-4 h per 1e9-step run on one GPU, 5 arms x 3 seeds):
    python run_icm_ablation.py
    python run_icm_ablation.py --steps 2e8 --arms ppo icm_mean icm_std   # shorter screen

CPU (debugging only: tiny envs/network, a few updates; checks that every arm
runs, logs the icm/* and exploration metrics and that the analysis works.
Its numbers say nothing about the hypothesis):
    python run_icm_ablation.py --preset cpu-debug

Runs are resumable: re-running the same command skips finished runs and resumes
interrupted ones from their checkpoints. Use --dry-run to print the commands.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARMS = {
    "ppo": {"algo.icm": "false"},
    "icm_none": {"algo.icm": "true", "algo.icm_normalize": "none"},
    "icm_std": {"algo.icm": "true", "algo.icm_normalize": "std"},
    "icm_mean": {"algo.icm": "true", "algo.icm_normalize": "mean"},
    "icm_ema": {"algo.icm": "true", "algo.icm_normalize": "ema"},
}

PRESETS = {
    # Full Craftax baseline config (configs/train_craftax.yaml) on an accelerator
    "gpu": {"steps": 1e9, "seeds": [0, 1, 2], "overrides": {"log_every": 10, "checkpoint_every": 50}},
    # Small enough to finish in minutes on a laptop CPU; for debugging the pipeline only
    "cpu-debug": {"steps": 32 * 16 * 12, "seeds": [0], "env": {"JAX_PLATFORMS": "cpu"}, "overrides": {
        "algo.num_envs": 32, "algo.num_steps": 16, "algo.num_minibatches": 2, "algo.update_epochs": 1,
        "algo.layer_size": 64, "algo.num_layers": 2, "algo.reset_ratio": 4,
        "log_every": 1, "checkpoint_every": 6}},
}


def build_runs(preset: str, arms, seeds, steps: float, out: Path, extra=()):
    p = PRESETS[preset]
    runs = []
    for arm in arms:
        for seed in seeds:
            run_dir = out / arm / f"seed{seed}"
            overrides = {**p["overrides"], **ARMS[arm], "seed": seed, "algo.total_timesteps": int(steps),
                         "output_dir": str(run_dir), "name": f"{arm}-seed{seed}"}
            cmd = [sys.executable, str(HERE / "train_craftax.py")] + [f"{k}={v}" for k, v in overrides.items()]
            runs.append({"arm": arm, "seed": seed, "dir": run_dir, "cmd": cmd + list(extra)})
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gpu")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, help="default: preset seeds")
    parser.add_argument("--steps", type=float, help="env steps per run (default: preset)")
    parser.add_argument("--out", default=None, help="default: runs/icm_ablation[_cpu_debug]")
    parser.add_argument("--wandb", action="store_true", help="also log every run to Weights & Biases")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()

    p = PRESETS[args.preset]
    out = Path(args.out or ("runs/icm_ablation" if args.preset == "gpu" else "runs/icm_ablation_cpu_debug"))
    seeds = args.seeds or p["seeds"]
    steps = args.steps or p["steps"]
    extra = ["wandb.enabled=true", "wandb.project=Craftax_ICM_ablation"] if args.wandb else []
    runs = build_runs(args.preset, args.arms, seeds, steps, out, extra)
    env = {**os.environ, **p.get("env", {})}
    print(f"[icm-ablation] preset={args.preset} arms={args.arms} seeds={seeds} steps={steps:.3g} -> {out}")
    if args.preset == "cpu-debug":
        print("[icm-ablation] CPU debug preset: checks the pipeline only; results do not test the hypothesis")

    for i, run in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}] {run['arm']} seed {run['seed']}"
        if (run["dir"] / "summary.json").exists():
            print(f"{tag}: finished, skipping")
            continue
        print(f"{tag}: {' '.join(run['cmd'][1:])}", flush=True)
        if not args.dry_run:
            subprocess.run(run["cmd"], cwd=HERE, env=env, check=True)

    if args.dry_run or args.no_analysis:
        return
    from analyze_icm_ablation import analyze, write_report
    result = analyze(out, args.arms)
    report = write_report(result, out, debug=args.preset == "cpu-debug")
    print(report)
    print(f"[icm-ablation] report: {out / 'report.md'}, data: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
