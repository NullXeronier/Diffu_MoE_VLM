"""
Train the MoE actor-critic on Crafter with PPO.

    python train_ppo.py                                   # defaults: configs/train_ppo.yaml
    python train_ppo.py ppo.total_steps=200000 model.encoder.name=vit
    python train_ppo.py model.trunk=mlp                   # dense ablation
"""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from diffu_moe_vlm.crafter_env import CRAFTER_ACTIONS, format_achievements
from diffu_moe_vlm.nn.policy import build_actor_critic
from diffu_moe_vlm.rl.checkpoint import save_checkpoint
from diffu_moe_vlm.rl.common import make_crafter_envs, make_logger, resolve_device, seed_everything, write_json
from diffu_moe_vlm.rl.ppo import PPOConfig, PPOTrainer
from diffu_moe_vlm.rl.rollout import ActorCriticRunner, evaluate


@hydra.main(config_path="configs", config_name="train_ppo", version_base=None)
def main(cfg) -> None:
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.get("torch_threads"))
    device = resolve_device(cfg.device)
    out = Path(cfg.output_dir)

    model_cfg = run_cfg['model']
    model = build_actor_critic(model_cfg, len(CRAFTER_ACTIONS), image_size=cfg.env.size)
    print(f"[PPO] parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")

    ppo_cfg = PPOConfig(**run_cfg['ppo'])
    envs = make_crafter_envs(ppo_cfg.num_envs, cfg.seed, cfg.env.size, cfg.env.length, cfg.env.num_workers)
    log = make_logger(run_cfg.get('wandb'), run_cfg, out / 'metrics.jsonl')
    trainer = PPOTrainer(
        model, envs, ppo_cfg, device=device, log_fn=log, checkpoint_every=cfg.checkpoint_every,
        checkpoint_fn=lambda step: save_checkpoint(out / 'checkpoint.pt', model, 'ppo', model_cfg,
                                                   trainer.optimizer, step),
    )
    try:
        tracker = trainer.train()
    finally:
        envs.close()
    path = save_checkpoint(out / 'checkpoint.pt', model, 'ppo', model_cfg, trainer.optimizer, trainer.global_step,
                           extra={'train_summary': tracker.summary(last=ppo_cfg.log_window)})
    print(f"[PPO] saved {path}")

    if cfg.eval_episodes > 0:
        eval_envs = make_crafter_envs(cfg.eval_num_envs, cfg.seed + 10_000, cfg.env.size, cfg.env.length,
                                      cfg.env.num_workers)
        try:
            result = evaluate(ActorCriticRunner(model.eval(), device), eval_envs, cfg.eval_episodes)
        finally:
            eval_envs.close()
        summary = result.summary()
        write_json(out / 'eval.json', summary)
        print(f"[Eval] episodes={summary['episodes']} return={summary['return_mean']:.2f} score={summary['score']:.2f}")
        print("[Eval] " + format_achievements(result.success_rates()))


if __name__ == '__main__':
    main()
