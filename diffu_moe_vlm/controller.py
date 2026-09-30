"""
Goal-conditioned controller.

Turns a sub-goal chosen by the selector (e.g. "obtain_stone_pickaxe") into
the next environment action. With `auto_prerequisites` enabled the controller
resolves missing tools and ingredients through the tech tree, like a scripted
skill library; disabled, it only attempts the goal itself and relies on the
planner to order sub-goals correctly (useful to measure planning quality).
"""

from typing import Dict, Optional

from .core import action_requirements, goal_item, obtain_method

Inventory = Dict[str, int]


class GoalController:
    """Scripted controller producing (verb, item) macro actions"""

    def __init__(self, auto_prerequisites: bool = True, max_depth: int = 16):
        self.auto_prerequisites = auto_prerequisites
        self.max_depth = max_depth

    def next_action(self, goal: str, inventory: Inventory) -> Optional[Dict[str, str]]:
        """
        Next action towards `goal`, or None when the goal item is unknown.

        Returns an action dict {'type': verb, 'item': item} accepted by the env.
        """
        item = goal_item(goal)
        verb = obtain_method(item)
        if verb is None:
            return None
        if not self.auto_prerequisites:
            return {'type': verb, 'item': item}
        return self._resolve(item, inventory, depth=0) or {'type': verb, 'item': item}

    def _resolve(self, item: str, inventory: Inventory, depth: int) -> Optional[Dict[str, str]]:
        """First executable action on the dependency path of one `item` action"""
        verb = obtain_method(item)
        if verb is None or depth > self.max_depth:
            return None
        tools, consumed = action_requirements(verb, item)
        for req, amount in {**tools, **consumed}.items():
            if inventory.get(req, 0) < amount:
                return self._resolve(req, inventory, depth + 1)
        return {'type': verb, 'item': item}
