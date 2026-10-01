"""
Evaluate a policy on Crafter and report achievement success rates and score.

    python evaluate_policy.py policy=random
    python evaluate_policy.py policy=runs/ppo/checkpoint.pt episodes=50
    python evaluate_policy.py policy=runs/diffusion/checkpoint.pt
"""

from pathlib import Path

import hydra

from diffu_moe_vlm.crafter_env import CRAFTER_ACTIONS, format_achievements
from diffu_moe_vlm.rl.checkpoint import load_policy
from diffu_moe_vlm.rl.common import make_crafter_envs, resolve_device, seed_everything, write_json
from diffu_moe_vlm.rl.rollout import ActorCriticRunner, DiffusionRunner, RandomRunner, evaluate


@hydra.main(config_path="configs", config_name="eval_policy", version_base=None)
def main(cfg) -> None:
    seed_everything(cfg.seed, cfg.get("torch_threads"))
    device = resolve_device(cfg.device)
    num_actions = len(CRAFTER_ACTIONS)
    if cfg.policy == 'random':
        runner = RandomRunner(num_actions, cfg.seed)
    else:
        model, ckpt = load_policy(cfg.policy, num_actions, device, cfg.env.size)
        if ckpt['kind'] == 'ppo':
            runner = ActorCriticRunner(model, device, deterministic=cfg.deterministic)
        else:
            runner = DiffusionRunner(model, device, cfg.execute_steps, cfg.sample_steps)

    envs = make_crafter_envs(cfg.num_envs, cfg.seed, cfg.env.size, cfg.env.length, cfg.env.num_workers)
    try:
        result = evaluate(runner, envs, cfg.episodes)
    finally:
        envs.close()
    summary = result.summary()
    out = Path(cfg.output_dir) / 'eval.json'
    write_json(out, {'policy': str(cfg.policy), **summary})
    print(f"[Eval] policy={cfg.policy} episodes={summary['episodes']} return={summary['return_mean']:.2f} "
          f"length={summary['length_mean']:.0f} score={summary['score']:.2f}")
    print("[Eval] " + format_achievements(result.success_rates()))
    print(f"[Eval] saved {out}")


if __name__ == '__main__':
    main()
