"""
Symbolic Minecraft core: items, tech tree, tasks and data loading.

This module is the single source of truth for recipes and task definitions.
The environment, controller, planner fallback and selector all use it, so a
change to the tech tree here is reflected everywhere.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"

Inventory = Dict[str, int]
Action = Tuple[str, str]  # (verb, item), verb in {"mine", "craft", "smelt"}


class MinecraftConstants:
    """Minecraft tech tree (simplified but consistent)"""

    # Raw resources gathered from the world: item -> required tool (None = bare hand)
    MINING = {
        'wood': None,
        'wool': None,
        'cobblestone': 'wooden_pickaxe',
        'coal': 'wooden_pickaxe',
        'iron_ore': 'stone_pickaxe',
        'diamond': 'iron_pickaxe',
    }

    # Crafting recipes: item -> ingredients (consumed), output quantity, crafting table required
    RECIPES = {
        'wooden_planks': {'ingredients': {'wood': 1}, 'output': 4, 'table': False},
        'stick': {'ingredients': {'wooden_planks': 2}, 'output': 4, 'table': False},
        'crafting_table': {'ingredients': {'wooden_planks': 4}, 'output': 1, 'table': False},
        'wooden_slab': {'ingredients': {'wooden_planks': 3}, 'output': 6, 'table': False},
        'wooden_pickaxe': {'ingredients': {'wooden_planks': 3, 'stick': 2}, 'output': 1, 'table': True},
        'stone_pickaxe': {'ingredients': {'cobblestone': 3, 'stick': 2}, 'output': 1, 'table': True},
        'iron_pickaxe': {'ingredients': {'iron_ingot': 3, 'stick': 2}, 'output': 1, 'table': True},
        'diamond_pickaxe': {'ingredients': {'diamond': 3, 'stick': 2}, 'output': 1, 'table': True},
        'furnace': {'ingredients': {'cobblestone': 8}, 'output': 1, 'table': True},
        'stone_stairs': {'ingredients': {'cobblestone': 6}, 'output': 4, 'table': True},
        'painting': {'ingredients': {'stick': 8, 'wool': 1}, 'output': 1, 'table': True},
    }

    # Smelting recipes: item -> ingredients (consumed). Requires a furnace and consumes FUEL.
    SMELTING = {
        'iron_ingot': {'ingredients': {'iron_ore': 1}, 'output': 1},
        'stone': {'ingredients': {'cobblestone': 1}, 'output': 1},
    }
    FUEL = {'coal': 1}

    ALL_ITEMS = sorted(set(MINING) | set(RECIPES) | set(SMELTING))
    ALL_CRAFT_SMELT_ITEMS = sorted(set(RECIPES) | set(SMELTING))

    BIOMES = ['Plains', 'Forest', 'Mountain', 'Desert', 'Swamp']


mc = MinecraftConstants()


# ---------------------------------------------------------------------------
# Tech tree helpers
# ---------------------------------------------------------------------------

def obtain_method(item: str) -> Optional[str]:
    """How an item is obtained: 'mine', 'craft', 'smelt' or None if unknown"""
    if item in mc.MINING:
        return 'mine'
    if item in mc.RECIPES:
        return 'craft'
    if item in mc.SMELTING:
        return 'smelt'
    return None


def action_requirements(verb: str, item: str) -> Tuple[Inventory, Inventory]:
    """
    Requirements for performing one (verb, item) action.

    Returns (tools, consumed): `tools` must be present but are not consumed
    (pickaxes, crafting table, furnace); `consumed` items are used up.
    """
    tools: Inventory = {}
    consumed: Inventory = {}
    if verb == 'mine':
        tool = mc.MINING.get(item)
        if tool:
            tools[tool] = 1
    elif verb == 'craft':
        recipe = mc.RECIPES[item]
        consumed.update(recipe['ingredients'])
        if recipe['table']:
            tools['crafting_table'] = 1
    elif verb == 'smelt':
        consumed.update(mc.SMELTING[item]['ingredients'])
        for fuel, amount in mc.FUEL.items():
            consumed[fuel] = consumed.get(fuel, 0) + amount
        tools['furnace'] = 1
    return tools, consumed


def output_quantity(verb: str, item: str) -> int:
    if verb == 'craft':
        return mc.RECIPES[item]['output']
    if verb == 'smelt':
        return mc.SMELTING[item]['output']
    return 1


def apply_action(inventory: Inventory, verb: str, item: str) -> Tuple[Inventory, bool, str]:
    """
    Apply a (verb, item) action to an inventory without mutating it.

    Returns (new_inventory, success, error_message).
    """
    if obtain_method(item) != verb:
        return dict(inventory), False, f"cannot {verb} {item}"

    tools, consumed = action_requirements(verb, item)
    missing = {
        req: amount - inventory.get(req, 0)
        for req, amount in {**tools, **consumed}.items()
        if inventory.get(req, 0) < amount
    }
    if missing:
        desc = ", ".join(f"{n} {k}" for k, n in missing.items())
        return dict(inventory), False, f"cannot {verb} {item}: missing {desc}"

    new_inv = dict(inventory)
    for req, amount in consumed.items():
        new_inv[req] -= amount
        if new_inv[req] == 0:
            del new_inv[req]
    new_inv[item] = new_inv.get(item, 0) + output_quantity(verb, item)
    return new_inv, True, ""


def plan_steps(target: str, inventory: Optional[Inventory] = None, quantity: int = 1,
               max_actions: int = 500) -> List[Action]:
    """
    Rule-based decomposition of a target into the ordered list of (verb, item)
    actions that obtains `quantity` of it from `inventory`, including repeated
    actions when more material is needed (e.g. six "mine cobblestone").
    Returns the actions found so far if the target is unreachable.
    """
    inv: Inventory = dict(inventory or {})
    steps: List[Action] = []

    def ensure(item: str, qty: int, depth: int) -> bool:
        verb = obtain_method(item)
        if verb is None or depth > 32:
            return inv.get(item, 0) >= qty
        while inv.get(item, 0) < qty:
            if len(steps) >= max_actions:
                return False
            tools, consumed = action_requirements(verb, item)
            requirements = {**tools, **consumed}
            # Gathering one requirement can consume another, so loop until all hold at once
            for _ in range(len(requirements) * 4 + 1):
                unmet = [(r, n) for r, n in requirements.items() if inv.get(r, 0) < n]
                if not unmet:
                    break
                if not ensure(unmet[0][0], unmet[0][1], depth + 1):
                    return False
            new_inv, ok, _ = apply_action(inv, verb, item)
            if not ok:
                return False
            inv.clear()
            inv.update(new_inv)
            steps.append((verb, item))
        return True

    ensure(target, quantity, 0)
    return steps


# ---------------------------------------------------------------------------
# Goal naming and natural-language parsing
# ---------------------------------------------------------------------------

def goal_name(item: str) -> str:
    """Canonical goal name for an item: mine_<raw item> or obtain_<crafted/smelted item>"""
    return f"mine_{item}" if obtain_method(item) == 'mine' else f"obtain_{item}"


def goal_item(goal: str) -> str:
    """Target item of a goal name (inverse of goal_name, tolerant of other prefixes)"""
    for prefix in ("mine_", "obtain_", "craft_", "smelt_"):
        if goal.startswith(prefix):
            return goal[len(prefix):]
    return goal


# Surface forms (lowercase, space separated) -> canonical item
ITEM_ALIASES: Dict[str, str] = {
    'log': 'wood', 'logs': 'wood', 'wood log': 'wood', 'wood logs': 'wood',
    'tree': 'wood', 'trees': 'wood', 'wood': 'wood', 'oak log': 'wood',
    'plank': 'wooden_planks', 'planks': 'wooden_planks',
    'sticks': 'stick', 'table': 'crafting_table', 'workbench': 'crafting_table',
    'sheep': 'wool', 'cobble': 'cobblestone', 'iron ingots': 'iron_ingot',
    'iron': 'iron_ore', 'coal ore': 'coal', 'diamonds': 'diamond',
    'slab': 'wooden_slab', 'slabs': 'wooden_slab', 'stairs': 'stone_stairs',
}
for _item in mc.ALL_ITEMS:
    _words = _item.replace('_', ' ')
    ITEM_ALIASES.setdefault(_words, _item)
    ITEM_ALIASES.setdefault(_words + 's', _item)

_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(ITEM_ALIASES, key=len, reverse=True)) + r")\b"
)
_MINE_VERBS = ('mine', 'dig', 'chop', 'collect', 'gather', 'find', 'get', 'harvest', 'kill', 'shear')


_COUNT_PATTERN = re.compile(r"(?:\(\s*)?[x\u00d7]\s*(\d+)\s*\)?\s*$")


def parse_plan_to_goals(text: str) -> List[str]:
    """
    Convert a free-form plan (one step per line) into canonical goal names.

    Each line contributes the first item it mentions, e.g.
    "Craft wooden planks from wood" -> obtain_wooden_planks. A trailing
    repeat count ("Mine cobblestone x6") repeats the goal. Plain "stone"
    with a gathering verb maps to cobblestone, as in Minecraft. Consecutive
    lines naming the same goal once ("Find trees", "Chop wood") are merged.
    """
    goals: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lower().replace('-', ' ')
        line = re.sub(r"^[\s\d\.\)\*]+", "", line)
        if not line:
            continue
        count = 1
        count_match = _COUNT_PATTERN.search(line)
        if count_match:
            count = max(1, int(count_match.group(1)))
            line = line[:count_match.start()]
        line_items = line.split(' from ')[0]  # "craft X from Y" -> X
        matches = _ALIAS_PATTERN.findall(line_items)
        if not matches:
            continue
        item = ITEM_ALIASES[matches[0]]
        if item == 'stone' and any(v in line_items for v in _MINE_VERBS):
            item = 'cobblestone'
        goal = goal_name(item)
        if count == 1 and goals and goals[-1] == goal:
            continue
        goals.extend([goal] * count)
    return goals


# ---------------------------------------------------------------------------
# Tasks and data
# ---------------------------------------------------------------------------

class TaskRegistry:
    """Benchmark task definitions"""

    def __init__(self):
        self.tasks = {
            'obtain_wooden_slab': {'target': 'wooden_slab', 'group': 'crafting', 'difficulty': 'easy'},
            'obtain_stone_stairs': {'target': 'stone_stairs', 'group': 'crafting', 'difficulty': 'medium'},
            'obtain_painting': {'target': 'painting', 'group': 'crafting', 'difficulty': 'medium'},
            'mine_cobblestone': {'target': 'cobblestone', 'group': 'mining', 'difficulty': 'easy'},
            'mine_iron_ore': {'target': 'iron_ore', 'group': 'mining', 'difficulty': 'medium'},
            'mine_diamond': {'target': 'diamond', 'group': 'mining', 'difficulty': 'hard'},
            'obtain_wooden_pickaxe': {'target': 'wooden_pickaxe', 'group': 'crafting', 'difficulty': 'easy'},
            'obtain_stone_pickaxe': {'target': 'stone_pickaxe', 'group': 'crafting', 'difficulty': 'medium'},
            'obtain_iron_pickaxe': {'target': 'iron_pickaxe', 'group': 'crafting', 'difficulty': 'hard'},
        }
        self.default_tasks = [
            'obtain_wooden_slab', 'obtain_stone_stairs', 'obtain_painting',
            'mine_cobblestone', 'mine_iron_ore', 'mine_diamond',
        ]

    def get_task(self, task_name: str) -> Optional[Dict]:
        """Get task definition (unknown obtain_/mine_ names resolve to their item)"""
        if task_name in self.tasks:
            return self.tasks[task_name]
        item = goal_item(task_name)
        if obtain_method(item):
            return {'target': item, 'group': 'crafting', 'difficulty': 'unknown'}
        return None

    def list_tasks(self) -> List[str]:
        """Default benchmark task list"""
        return list(self.default_tasks)


class DataManager:
    """Loads packaged data files (goal library, mappings, task info, prompts)"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def _load_json(self, filename: str) -> Dict:
        path = self.data_dir / filename
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def load_goal_mapping(self) -> Dict:
        return self._load_json("goal_mapping.json")

    def load_task_info(self) -> Dict:
        return self._load_json("task_info.json")

    def load_text(self, filename: str) -> Optional[str]:
        path = self.data_dir / filename
        return path.read_text() if path.exists() else None

    def save_data(self, filename: str, data: Dict):
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class SimulationState:
    """Inventory-only simulation state backed by the tech tree"""

    def __init__(self, inventory: Optional[Inventory] = None):
        self.inventory: Inventory = dict(inventory or {})

    def get_inventory(self) -> Inventory:
        return dict(self.inventory)

    def can_perform(self, verb: str, item: str) -> bool:
        return apply_action(self.inventory, verb, item)[1]

    def perform(self, verb: str, item: str) -> Tuple[bool, str]:
        self.inventory, success, error = apply_action(self.inventory, verb, item)
        return success, error


# Global instances
task_registry = TaskRegistry()
data_manager = DataManager()
