"""
Compare a Craftax run with the reference W&B results (docs/craftax_reference.json).

    python compare_craftax_run.py runs/craftax/metrics.jsonl --ref ppo
    python compare_craftax_run.py runs/craftax/metrics.jsonl --ref ppo_rnn --last 0.05

Metrics are averaged over the last `--last` fraction of logged updates. The
reference is the end of a 1e9-step run, so only full-length runs are comparable;
for shorter runs the table shows how far along the run is.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare(rows, reference, last: float = 0.05):
    n = max(1, int(round(len(rows) * last)))
    tail = rows[-n:]
    results = []
    for key, (lo, hi) in reference["metrics"].items():
        values = [r[key] for r in tail if key in r]
        if not values:
            results.append((key, None, lo, hi, "missing"))
            continue
        value = float(np.mean(values))
        status = "ok" if lo <= value <= hi else ("low" if value < lo else "high")
        results.append((key, value, lo, hi, status))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", help="metrics.jsonl written by train_craftax.py")
    parser.add_argument("--ref", default="ppo", help="reference run: ppo | ppo_rnn")
    parser.add_argument("--reference-file", default=str(Path(__file__).parent / "docs" / "craftax_reference.json"))
    parser.add_argument("--last", type=float, default=0.05, help="fraction of logged updates to average")
    args = parser.parse_args()

    ref_all = json.loads(Path(args.reference_file).read_text())
    reference = ref_all["runs"][args.ref]
    rows = load_metrics(args.metrics)
    steps = rows[-1].get("env_steps", 0)
    print(f"run: {args.metrics} ({len(rows)} logged updates, {steps:.3g} env steps)")
    print(f"reference: {args.ref} - {reference['description']} at {ref_all['total_timesteps']:.0e} steps")
    if steps < 0.95 * ref_all["total_timesteps"]:
        print(f"NOTE: run has {steps / ref_all['total_timesteps']:.1%} of the reference budget; "
              "'low' rows are expected until it is complete")
    results = compare(rows, reference, args.last)
    counts = {}
    for key, value, lo, hi, status in results:
        counts[status] = counts.get(status, 0) + 1
        shown = "-" if value is None else f"{value:8.2f}"
        print(f"{key:40s} {shown:>8s}   ref [{lo:g}, {hi:g}]   {status}")
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
