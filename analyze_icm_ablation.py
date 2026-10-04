"""
Analyze an ICM-normalization ablation written by run_icm_ablation.py.

    python analyze_icm_ablation.py runs/icm_ablation

Per run (metrics.jsonl of train_craftax.py), with `tail` = the last 10% of logged updates:
    final_return      extrinsic episode return over the tail (the bonus is not included)
    auc_return        extrinsic return averaged over the whole run (learning speed)
    final_score       Craftax score (geometric mean of achievement rates) over the tail
    final_achievements, final_coverage   achievements per episode / achievement types reached
    coverage_ever     achievement types reached at least once during the run (exploration)
    bonus_share_*     |bonus| / (|bonus| + |extrinsic|) per update: max, tail mean, and the
                      fraction of updates above RUNAWAY_SHARE
    bonus_growth      tail / head (first 10%) mean bonus per step
    runaway           the bonus still dominates the reward at the end of training (tail share >
                      RUNAWAY_SHARE) or grew more than RUNAWAY_GROWTH times. Early updates are not
                      used: the extrinsic reward is close to zero there, so even a well-scaled bonus
                      is most of the reward
    raw_error_growth  last / first raw forward-model error (does the unnormalized error grow?)

Questions answered in report.md (paired by seed against the `ppo` baseline):
    H1  Is the icm_mean bonus bounded, while unnormalized / std-normalized bonuses run away?
    H2  Does icm_mean explore better than ppo (coverage_ever, final_score) without hurting
        the extrinsic return? Needs >= 3 seeds and the same sign on every seed to count.
"""

import argparse
import json
import math
from pathlib import Path

RUNAWAY_SHARE = 0.5   # the curiosity bonus is more than half of the total |reward|
RUNAWAY_GROWTH = 2.0  # the mean bonus per step at the end is more than twice that at the start
TAIL = 0.1
MIN_VERDICT_STEPS = 5e7  # shorter runs (e.g. the CPU debug preset) only get "too short" verdicts
ARM_ORDER = ["ppo", "icm_none", "icm_std", "icm_mean", "icm_ema"]
MIN_SEEDS = 3
RUN_METRICS = ["final_return", "auc_return", "final_score", "final_achievements", "final_coverage",
               "coverage_ever", "bonus_share_max", "bonus_share_final", "runaway_fraction", "bonus_growth", "runaway",
               "raw_error_growth"]


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_stats(rows):
    """Summary numbers of one run (see module docstring)"""
    ep_rows = [r for r in rows if r.get("episodes", 0) > 0] or rows
    tail = ep_rows[-max(1, int(round(len(ep_rows) * TAIL))):]
    ach_keys = sorted({k for r in rows for k in r if k.startswith("Achievements/")})
    s = {
        "updates": rows[-1].get("update") if rows else 0,
        "env_steps": rows[-1].get("env_steps") if rows else 0,
        "final_return": _mean([r.get("episode_return") for r in tail]),
        "auc_return": _mean([r.get("episode_return") for r in ep_rows]),
        "final_score": _mean([r.get("craftax_score") for r in tail]),
        "final_achievements": _mean([r.get("achievements") for r in tail]),
        "final_coverage": _mean([r.get("achievement_coverage") for r in tail]),
        "coverage_ever": sum(1 for k in ach_keys if max(r.get(k, 0.0) for r in rows) > 0),
        "achievement_types": len(ach_keys),
    }
    shares = [r["icm/bonus_share"] for r in rows if "icm/bonus_share" in r]
    if shares:
        raw = [r["icm/raw_error_mean"] for r in rows if "icm/raw_error_mean" in r]
        bonus = [r["icm/bonus_mean"] for r in rows if "icm/bonus_mean" in r]
        n = max(1, int(round(len(shares) * TAIL)))
        head, end = _mean(bonus[:n]), _mean(bonus[-n:])
        growth = end / head if head and head > 0 else None
        share_final = _mean(shares[-n:])
        s.update({
            "bonus_share_max": max(shares),
            "bonus_share_final": share_final,
            "runaway_fraction": sum(x > RUNAWAY_SHARE for x in shares) / len(shares),
            "bonus_growth": growth,
            "runaway": 1.0 if share_final > RUNAWAY_SHARE or (growth is not None and growth > RUNAWAY_GROWTH) else 0.0,
            "raw_error_growth": raw[-1] / raw[0] if raw and raw[0] > 0 else None,
            "bonus_mean_final": rows[-1].get("icm/bonus_mean"),
            "bonus_max": max(r.get("icm/bonus_max", 0.0) for r in rows),
        })
    return s


def analyze(out_dir, arms=None):
    out_dir = Path(out_dir)
    if not arms:
        found = [p.name for p in out_dir.iterdir() if p.is_dir()]
        arms = [a for a in ARM_ORDER if a in found] + sorted(a for a in found if a not in ARM_ORDER)
    result = {"arms": {}, "runaway_share": RUNAWAY_SHARE}
    for arm in arms:
        runs = {}
        for f in sorted((out_dir / arm).glob("seed*/metrics.jsonl")):
            rows = load_rows(f)
            if rows:
                runs[int(f.parent.name[4:])] = run_stats(rows)
        if not runs:
            continue
        agg = {k: {"mean": _mean([r.get(k) for r in runs.values()]), "std": _std([r.get(k) for r in runs.values()
                                                                                if r.get(k) is not None])}
               for k in RUN_METRICS if any(r.get(k) is not None for r in runs.values())}
        result["arms"][arm] = {"seeds": runs, "aggregate": agg}
    result["comparisons"] = {arm: paired(result, arm, "ppo") for arm in result["arms"] if arm != "ppo"}
    result["verdict"] = verdict(result)
    return result


def paired(result, arm, base):
    """Per-seed differences arm - base for the exploration and return metrics"""
    a, b = result["arms"].get(arm), result["arms"].get(base)
    if not a or not b:
        return {}
    seeds = sorted(set(a["seeds"]) & set(b["seeds"]))
    out = {}
    for k in ["coverage_ever", "final_score", "final_achievements", "final_return", "auc_return"]:
        diffs = [a["seeds"][s][k] - b["seeds"][s][k] for s in seeds
                 if a["seeds"][s].get(k) is not None and b["seeds"][s].get(k) is not None]
        if diffs:
            out[k] = {"mean_diff": _mean(diffs), "std_diff": _std(diffs), "n": len(diffs),
                      "positive": sum(d > 0 for d in diffs), "negative": sum(d < 0 for d in diffs)}
    return out


def _bounded(result, arm):
    runs = result["arms"].get(arm, {}).get("seeds", {})
    if not runs or any("runaway" not in r for r in runs.values()):
        return None
    return all(r["runaway"] == 0.0 for r in runs.values())


def verdict(result):
    v = {}
    steps = [r["env_steps"] for a in result["arms"].values() for r in a["seeds"].values()]
    if steps and min(steps) < MIN_VERDICT_STEPS:
        why = f"runs are too short ({min(steps):.3g} env steps < {MIN_VERDICT_STEPS:.0e}) to judge"
        return {"H1": ("too short", why), "H2": ("too short", why)}
    mean_ok = _bounded(result, "icm_mean")
    controls = {arm: _bounded(result, arm) for arm in ("icm_none", "icm_std") if arm in result["arms"]}
    ran_away = [arm for arm, ok in controls.items() if ok is False]
    if mean_ok is None:
        v["H1"] = ("missing", "no icm_mean runs")
    elif not mean_ok:
        v["H1"] = ("not supported", "the icm_mean bonus ran away (dominated the reward at the end or grew) on at least one seed")
    elif ran_away:
        v["H1"] = ("supported", f"icm_mean stayed bounded on every seed while {', '.join(ran_away)} ran away")
    else:
        v["H1"] = ("inconclusive", "icm_mean stayed bounded, but no control arm ran away in this budget")

    c = result["comparisons"].get("icm_mean", {})
    if not c:
        v["H2"] = ("missing", "needs both ppo and icm_mean runs")
    else:
        n = max(x["n"] for x in c.values())
        explore = [c[k] for k in ("coverage_ever", "final_score") if k in c]
        all_up = any(x["positive"] == x["n"] for x in explore)
        any_down = any(x["mean_diff"] < 0 for x in explore)
        ret = c.get("final_return")
        hurts = ret is not None and ret["negative"] == ret["n"]
        if n < MIN_SEEDS:
            v["H2"] = ("inconclusive", f"only {n} paired seed(s); need >= {MIN_SEEDS}")
        elif all_up and not any_down and not hurts:
            v["H2"] = ("supported", "coverage or score higher than ppo on every seed, extrinsic return not lower on all seeds")
        elif any_down and not all_up:
            v["H2"] = ("not supported", "exploration metrics are lower than ppo on average")
        else:
            v["H2"] = ("inconclusive", "mixed signs across seeds or extrinsic return lower on every seed")
    return v


def _fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(result, out_dir, debug=False):
    out_dir = Path(out_dir)
    lines = ["# ICM normalization ablation", ""]
    if debug:
        lines += ["> CPU debug preset: tiny model and a few updates. This checks the pipeline only;",
                  "> the numbers below do not test the hypothesis.", ""]
    lines += ["Mean ± std over seeds. Return is extrinsic only. Runaway = over the last "
              f"{TAIL:.0%} of updates the bonus is > {RUNAWAY_SHARE:.0%} of |reward|, or the mean bonus grew "
              f"> {RUNAWAY_GROWTH:g}x; 'runaway' is the fraction of seeds.", "",
              "| arm | seeds | final return | AUC return | Craftax score | coverage ever | bonus share (end) | bonus growth | runaway | raw error growth |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, data in result["arms"].items():
        a = data["aggregate"]
        cell = lambda k, nd=2: (f"{_fmt(a[k]['mean'], nd)} ± {_fmt(a[k]['std'], nd)}" if k in a else "-")
        lines.append(f"| {arm} | {len(data['seeds'])} | {cell('final_return')} | {cell('auc_return')} | "
                     f"{cell('final_score')} | {cell('coverage_ever', 1)} | {cell('bonus_share_final')} | "
                     f"{cell('bonus_growth')} | {cell('runaway')} | {cell('raw_error_growth')} |")
    lines += ["", "Paired differences vs. ppo (same seeds): mean diff (seeds up / down)", "",
              "| arm | coverage ever | Craftax score | achievements | final return | AUC return |", "|---|---|---|---|---|---|"]
    for arm, c in result["comparisons"].items():
        cell = lambda k: (f"{c[k]['mean_diff']:+.2f} ({c[k]['positive']}↑/{c[k]['negative']}↓)" if k in c else "-")
        lines.append(f"| {arm} | {cell('coverage_ever')} | {cell('final_score')} | {cell('final_achievements')} | "
                     f"{cell('final_return')} | {cell('auc_return')} |")
    lines += ["", "## Verdict", ""]
    names = {"H1": "Moving-mean normalization keeps the curiosity bonus bounded",
             "H2": "Moving-mean normalization improves exploration over PPO"}
    for h, (status, why) in result["verdict"].items():
        lines.append(f"- **{h}** {names[h]}: **{status}** ({why})")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", help="directory written by run_icm_ablation.py")
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    result = analyze(args.out_dir, args.arms)
    print(write_report(result, args.out_dir))


if __name__ == "__main__":
    main()
