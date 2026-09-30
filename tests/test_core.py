import pytest

from diffu_moe_vlm.core import (
    apply_action, goal_item, goal_name, mc, obtain_method, parse_plan_to_goals, plan_steps, task_registry,
)
from diffu_moe_vlm.planner import Planner


def simulate(steps, inventory=None):
    inventory = dict(inventory or {})
    for verb, item in steps:
        inventory, ok, error = apply_action(inventory, verb, item)
        assert ok, error
    return inventory


def test_apply_action_consumes_ingredients_and_requires_tools():
    inv, ok, _ = apply_action({'wood': 1}, 'craft', 'wooden_planks')
    assert ok and inv == {'wooden_planks': 4}

    inv, ok, error = apply_action({}, 'mine', 'cobblestone')
    assert not ok and 'wooden_pickaxe' in error

    inv, ok, _ = apply_action({'wooden_pickaxe': 1}, 'mine', 'cobblestone')
    assert ok and inv == {'wooden_pickaxe': 1, 'cobblestone': 1}  # tools are not consumed


def test_apply_action_rejects_wrong_verb():
    _, ok, _ = apply_action({'wood': 5}, 'mine', 'wooden_planks')
    assert not ok


@pytest.mark.parametrize("task", sorted(task_registry.tasks))
def test_plan_steps_reach_every_task_target(task):
    target = task_registry.get_task(task)['target']
    final = simulate(plan_steps(target))
    assert final.get(target, 0) >= 1


def test_plan_steps_respects_quantities_and_existing_inventory():
    steps = plan_steps('stone_stairs')
    assert steps.count(('mine', 'cobblestone')) == 6
    # Two planks are not enough for a pickaxe (needs 3), so more wood is gathered
    steps = plan_steps('wooden_pickaxe', {'crafting_table': 1, 'wooden_planks': 2, 'stick': 4})
    assert steps == [('mine', 'wood'), ('craft', 'wooden_planks'), ('craft', 'wooden_pickaxe')]


def test_goal_names_roundtrip():
    for item in mc.ALL_ITEMS:
        assert goal_item(goal_name(item)) == item
    assert goal_name('wood') == 'mine_wood'
    assert goal_name('iron_ingot') == 'obtain_iron_ingot'
    assert obtain_method('unknown_item') is None


def test_parse_plan_to_goals_handles_llm_style_text():
    plan = (
        "1. Find and approach trees\n"
        "2. Mine wood logs from trees\n"
        "3. Craft wooden planks from wood (4 planks per log)\n"
        "4. Craft a crafting table\n"
        "5. Craft a stone pickaxe\n"
        "6. Mine stone x3\n"
        "7. Look around\n"
    )
    assert parse_plan_to_goals(plan) == [
        'mine_wood', 'obtain_wooden_planks', 'obtain_crafting_table', 'obtain_stone_pickaxe',
        'mine_cobblestone', 'mine_cobblestone', 'mine_cobblestone',
    ]


def test_fallback_plan_roundtrips_through_parser():
    steps = plan_steps('iron_pickaxe')
    goals = parse_plan_to_goals(Planner.format_plan(steps))
    assert [goal_item(g) for g in goals] == [item for _, item in steps]
