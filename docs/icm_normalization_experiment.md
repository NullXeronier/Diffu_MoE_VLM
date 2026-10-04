# Experiment: ICM bonus normalization

**Question.** Does dividing the ICM curiosity bonus by a moving mean of the forward-model error
(1) stop the intrinsic reward from running away and (2) actually improve exploration on Craftax?

Earlier evidence is only anecdotal: dividing by the standard deviation left the bonus with a mean
of about 12 (it swamped the extrinsic reward), and the reference ICM run in W&B collapsed to near
zero extrinsic reward. Neither was a controlled comparison. This experiment is one.

## Design

Same Craftax-Symbolic-v1 PPO (baseline config, `configs/train_craftax.yaml`), same seeds, and the
same `icm_reward_coef = 0.01` in every ICM arm, so the only difference is the normalization:

| Arm | Bonus per step | Expectation |
|---|---|---|
| `ppo` | none | baseline |
| `icm_none` | `0.01 · e` | runs away when the raw error is large |
| `icm_std` | `0.01 · e / std(e)` | still too large: squared errors are mostly mean |
| `icm_mean` | `0.01 · e / mean(e)` (running mean, default) | bounded, averages 0.01 |
| `icm_ema` | `0.01 · e / EMA(e)` (decay 0.99 per update) | bounded and follows a shrinking error |

`e` is the ICM forward-model error of each transition.

## Metrics (logged every update by `train_craftax.py`)

- **Runaway:** `icm/bonus_share` = |bonus| / (|bonus| + |extrinsic reward|), plus `icm/bonus_mean`.
  - A run counts as runaway if, over the last 10% of updates, the share is above 0.5, or if the mean bonus per step grew more than 2× from the first 10%.
  - Early updates are not used. The extrinsic reward is close to zero there, so even a correctly scaled bonus is most of the reward.
  - Also logged: `icm/raw_error_mean`, `icm/raw_error_max`, `icm/scale` and `icm/bonus_max`.
- **Exploration:**
  - `achievement_coverage`: achievement types reached in the update.
  - `craftax_score`: the geometric mean of achievement success rates.
  - `achievements` per episode.
  - Offline, `coverage_ever`: achievement types reached at least once during the run.
- **Task performance:** `episode_return`, which is extrinsic only, so arms are compared on the real reward.

## Decision rules (`analyze_icm_ablation.py`)

- Runs shorter than 5·10⁷ steps get no verdict ("too short").
- **H1 (bounded):**
  - *Supported* if `icm_mean` does not run away on any seed, while `icm_none` or `icm_std` does.
  - *Not supported* if `icm_mean` runs away on any seed.
  - *Inconclusive* if no control arm runs away within the budget.
- **H2 (better exploration):**
  - *Supported* if, compared with `ppo` on the same seed, `coverage_ever` or `final_score` is higher on every seed (at least 3 seeds), and the extrinsic return is not lower on every seed.
  - *Not supported* if the exploration metrics are lower on average.
  - *Inconclusive* otherwise.

## Running it

GPU (the experiment). This is 5 arms × 3 seeds × 1e9 steps, about 3–4 hours per run on one GPU. A shorter screen is also available:

```bash
pip install -e ".[jax]" && pip install -U "jax[cuda12]"
python run_icm_ablation.py
python run_icm_ablation.py --steps 2e8 --arms ppo icm_mean icm_std
```

Results go to `runs/icm_ablation/report.md` and `summary.json`. Re-running the command skips
finished runs and resumes interrupted ones from their checkpoints. `--wandb` also logs every run to W&B.

CPU (debugging only). This uses 32 envs × 16 steps, a 64-wide network, 12 updates per arm and one seed.
It checks that every arm runs, that the metrics are logged and that the report is produced:

```bash
python run_icm_ablation.py --preset cpu-debug
```

The CPU debug numbers do not test either hypothesis, and the report says "too short" instead of giving a verdict.
With 12 tiny updates the policies barely move, so the arms sample almost the same actions and show the same extrinsic return.

## Status

- The code is implemented and unit-tested (`tests/test_jax_rl.py`). It covers the normalization modes, the bonus metrics, runaway detection, paired comparison and the runner arms.
- The CPU debug preset runs end to end: all 5 arms, the icm/* and exploration metrics, and the report.
- The GPU experiment has **not been run yet**, so neither hypothesis has an answer.
