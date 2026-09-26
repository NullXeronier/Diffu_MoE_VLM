import pytest
from gymnasium.utils.env_checker import check_env

from diffu_moe_vlm.controller import GoalController
from diffu_moe_vlm.core import goal_name, task_registry
from diffu_moe_vlm.env import MinecraftGymnasiumEnv


def make_env(**kwargs):
    return MinecraftGymnasiumEnv(img_size=(32, 24), **kwargs)


def test_passes_gymnasium_env_checker():
    check_env(make_env(), skip_render_check=True)


def test_observation_layout():
    env = make_env()
    obs, info = env.reset(options={'task': 'obtain_wooden_slab'})
    assert obs['rgb'].shape == (24, 32, 3)
    assert obs['inventory'].sum() == 0
    assert info['target'] == 'wooden_slab' and info['inventory'] == {}


def test_actions_change_inventory_and_succeed_on_target():
    env = make_env()
    env.reset(options={'task': 'obtain_wooden_slab'})
    _, _, _, _, info = env.step({'type': 'craft', 'item': 'wooden_planks'})
    assert not info['action_success'] and 'wood' in info['action_error']

    env.step(('mine', 'wood'))
    env.step(env.encode_action('craft', 'wooden_planks'))
    _, reward, terminated, truncated, info = env.step({'type': 'craft', 'item': 'wooden_slab'})
    assert info['inventory']['wooden_slab'] == 6
    assert reward == 1.0 and terminated and not truncated and info['task_success']


def test_truncates_at_max_steps():
    env = make_env(max_steps=2)
    env.reset(options={'task': 'mine_diamond'})
    assert not env.step(0)[3]
    assert env.step(0)[3]


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        make_env().reset(options={'task': 'obtain_unobtainium'})


@pytest.mark.parametrize("task", sorted(task_registry.tasks))
def test_controller_with_prerequisites_reaches_every_target(task):
    env = make_env(max_steps=200)
    _, info = env.reset(options={'task': task})
    controller = GoalController(auto_prerequisites=True)
    goal = goal_name(info['target'])
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(controller.next_action(goal, info['inventory']))
        assert info['action_success'], info['action_error']
        if terminated or truncated:
            break
    assert info['task_success']


def test_controller_without_prerequisites_only_attempts_goal():
    controller = GoalController(auto_prerequisites=False)
    assert controller.next_action('obtain_iron_pickaxe', {}) == {'type': 'craft', 'item': 'iron_pickaxe'}
    assert controller.next_action('obtain_nothing', {}) is None
