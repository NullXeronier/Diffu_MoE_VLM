"""
Behavior-clone a diffusion policy on Crafter.

Demonstrations come from an existing .npz (`demos.path`) or are collected with
a PPO checkpoint (`demos.policy_checkpoint`); without either, a random policy
is used (only useful as a smoke test).

    python train_diffusion.py demos.policy_checkpoint=runs/ppo/checkpoint.pt
"""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from diffu_moe_vlm.crafter_env import CRAFTER_ACTIONS, format_achievements
from diffu_moe_vlm.nn.diffusion import build_diffusion_policy
from diffu_moe_vlm.rl.checkpoint import load_policy, save_checkpoint
from diffu_moe_vlm.rl.common import make_crafter_envs, make_logger, resolve_device, seed_everything, write_json
from diffu_moe_vlm.rl.diffusion_bc import train_diffusion_bc
from diffu_moe_vlm.rl.rollout import ActorCriticRunner, DiffusionRunner, RandomRunner, collect_demonstrations, evaluate


@hydra.main(config_path="configs", config_name="train_diffusion", version_base=None)
def main(cfg) -> None:
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.get("torch_threads"))
    device = resolve_device(cfg.device)
    out = Path(cfg.output_dir)
    num_actions = len(CRAFTER_ACTIONS)

    # 1. Demonstrations
    if cfg.demos.path and Path(cfg.demos.path).exists():
        with np.load(cfg.demos.path) as f:
            data = {k: f[k] for k in f.files}
        print(f"[Demos] loaded {len(data['actions'])} steps from {cfg.demos.path}")
    else:
        if cfg.demos.policy_checkpoint:
            teacher, _ = load_policy(cfg.demos.policy_checkpoint, num_actions, device, cfg.env.size)
            runner = ActorCriticRunner(teacher, device)
            print(f"[Demos] collecting with {cfg.demos.policy_checkpoint}")
        else:
            runner = RandomRunner(num_actions, cfg.seed)
            print("[Demos] no teacher checkpoint given: collecting RANDOM demonstrations (smoke test only)")
        envs = make_crafter_envs(cfg.demos.num_envs, cfg.seed, cfg.env.size, cfg.env.length, cfg.env.num_workers)
        try:
            data = collect_demonstrations(runner, envs, cfg.demos.num_steps, path=out / 'demos.npz')
        finally:
            envs.close()
        print(f"[Demos] collected {len(data['actions'])} steps in {data['episode'].max() + 1} episodes")

    # 2. Behavior cloning
    model_cfg = run_cfg['model']
    model = build_diffusion_policy(model_cfg, num_actions, image_size=cfg.env.size)
    log = make_logger(run_cfg.get('wandb'), run_cfg, out / 'metrics.jsonl')
    train_diffusion_bc(model, data, steps=cfg.train.steps, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                       device=device, log_fn=log, seed=cfg.seed)
    path = save_checkpoint(out / 'checkpoint.pt', model, 'diffusion', model_cfg, step=cfg.train.steps)
    print(f"[Diffusion BC] saved {path}")

    # 3. Evaluation
    if cfg.eval.episodes > 0:
        envs = make_crafter_envs(cfg.eval.num_envs, cfg.seed + 10_000, cfg.env.size, cfg.env.length,
                                 cfg.env.num_workers)
        runner = DiffusionRunner(model, device, cfg.eval.execute_steps, cfg.eval.sample_steps)
        try:
            result = evaluate(runner, envs, cfg.eval.episodes)
        finally:
            envs.close()
        summary = result.summary()
        write_json(out / 'eval.json', summary)
        print(f"[Eval] episodes={summary['episodes']} return={summary['return_mean']:.2f} score={summary['score']:.2f}")
        print("[Eval] " + format_achievements(result.success_rates()))


if __name__ == '__main__':
    main()
