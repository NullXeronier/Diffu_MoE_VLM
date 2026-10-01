"""
PPO / PPO-RNN on Craftax in JAX (runs end-to-end on one GPU).

    python train_craftax.py                                  # PPO, Craftax-Symbolic-v1, 1B steps
    python train_craftax.py algo.rnn=true                    # PPO-RNN
    python train_craftax.py algo.moe=true                    # MoE hidden layers
    python train_craftax.py algo.icm=true                    # + normalized ICM bonus
    python train_craftax.py algo.total_timesteps=10000000 algo.num_envs=256   # small CPU run

Writes metrics.jsonl (one line per logged update), checkpoint.npz (full
training state, written every `checkpoint_every` updates), params.msgpack and
summary.json to output_dir. Re-running with the same output_dir resumes from
the checkpoint with identical results, so preempted or crashed runs can be
continued; `max_updates_per_run` stops early on purpose (time-limited jobs).
Compare a finished run with the reference W&B results using
compare_craftax_run.py.
"""

import json
import time
from pathlib import Path

import hydra
import jax
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from diffu_moe_vlm.jax_rl.ppo import load_runner, make_train, save_runner


def run_name(cfg) -> str:
    algo = cfg["algo"]
    tag = "PPO_RNN-" if algo["rnn"] else ""
    tag += "MoE-" if algo["moe"] else ""
    tag += "ICM-" if algo["icm"] else ""
    return f"{algo['env_name']}-{tag}{int(algo['total_timesteps'] // 1_000_000)}M"


@hydra.main(config_path="configs", config_name="train_craftax", version_base=None)
def main(cfg) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    name = cfg.get("name") or run_name(cfg)
    print(f"[Craftax] {name} on {jax.devices()}")
    print(json.dumps(cfg["algo"], indent=2))

    wandb_run = None
    if cfg["wandb"]["enabled"]:
        import wandb
        wandb_run = wandb.init(project=cfg["wandb"]["project"], name=name, config=cfg["algo"])

    start = time.time()
    state = {"last_step": None, "last_time": start}
    jsonl = None

    def log_fn(metrics, update_idx):
        update = int(update_idx) + 1
        if update % cfg["log_every"] and update != train.num_updates:
            return
        m = {k: float(np.asarray(v)) for k, v in metrics.items()}
        now = time.time()
        if state["last_step"] is not None:
            m["sps"] = (m["env_steps"] - state["last_step"]) / max(now - state["last_time"], 1e-9)
        state.update(last_step=m["env_steps"], last_time=now)
        jsonl.write(json.dumps(m) + "\n")
        jsonl.flush()
        if wandb_run:
            wandb_run.log(m, step=update)
        print(f"[Craftax] update {update}/{train.num_updates} steps={m['env_steps']:.3g} sps={m.get('sps', 0):.0f} "
              f"return={m['episode_return']:.2f} length={m['episode_length']:.0f} episodes={m['episodes']:.0f}",
              flush=True)

    train = make_train(cfg["algo"], log_fn=log_fn)
    ckpt_path = out / "checkpoint.npz"
    runner = train.init(jax.random.PRNGKey(cfg["seed"]))
    next_update = 0
    if cfg["resume"] and ckpt_path.exists():
        runner, next_update = load_runner(ckpt_path, runner)
        print(f"[Craftax] resumed from {ckpt_path} at update {next_update}/{train.num_updates}")
    jsonl = open(out / "metrics.jsonl", "a" if next_update else "w")

    budget = cfg.get("max_updates_per_run") or train.num_updates
    stop_at = min(train.num_updates, next_update + budget)
    chunk = max(1, int(cfg["checkpoint_every"]))
    while next_update < stop_at:
        num = min(chunk, stop_at - next_update)
        runner, _ = jax.block_until_ready(train.run_updates(runner, next_update, num))
        next_update += num
        save_runner(ckpt_path, runner, next_update)
    jsonl.close()

    if next_update < train.num_updates:
        print(f"[Craftax] stopped at update {next_update}/{train.num_updates}; re-run to resume from {ckpt_path}")
        return
    params = runner[0].params
    (out / "params.msgpack").write_bytes(serialization.to_bytes(params))
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
    summary = {"name": name, "seed": cfg["seed"], "final": rows[-1] if rows else {}, "config": train.config}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Craftax] finished {train.num_updates} updates; saved to {out}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
