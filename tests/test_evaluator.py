from pathlib import Path

import pytest
from omegaconf import OmegaConf

from diffu_moe_vlm.evaluator import Evaluator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def cfg(tmp_path):
    cfg = OmegaConf.load(CONFIG_DIR / "defaults.yaml")
    for group in ("data/planning", "eval/planning", "goal_model/horizon"):
        name = group.split("/")[0]
        cfg[name] = OmegaConf.merge(OmegaConf.load(CONFIG_DIR / f"{group}.yaml"), cfg.get(name, {}))
    cfg.wandb.enabled = False
    cfg.llm.enabled = False
    cfg.output_dir = str(tmp_path)
    cfg.simulator.resolution = [32, 24]
    return cfg


def test_rule_based_baseline_solves_default_tasks(cfg):
    results = Evaluator(cfg).run_all_tasks()
    assert len(results) == 6
    assert all(r['success'] for r in results), results


@pytest.mark.parametrize("auto_prerequisites", [True, False])
def test_replanning_recovers_from_random_action_failures(cfg, auto_prerequisites):
    cfg.env.action_failure_prob = 0.3
    cfg.controller.auto_prerequisites = auto_prerequisites
    result = Evaluator(cfg).single_task_evaluate('obtain_stone_stairs')
    assert result['success']


def test_bad_plan_needs_controller_prerequisites(cfg):
    """An LLM plan that skips steps only works when the controller fills the gaps"""
    outcomes = {}
    for auto in (True, False):
        cfg.controller.auto_prerequisites = auto
        evaluator = Evaluator(cfg)
        evaluator.planner.llm_available = True
        evaluator.planner.query_llm = lambda prompt: "1. Craft a wooden pickaxe\n2. Mine cobblestone"
        outcomes[auto] = evaluator.single_task_evaluate('mine_cobblestone')
    assert outcomes[True]['success']
    assert not outcomes[False]['success']
    assert outcomes[False]['replans'] == cfg.eval.max_replans
