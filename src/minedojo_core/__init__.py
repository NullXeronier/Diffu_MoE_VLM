"""
MineDojo Core Components
Reconstructed data, sim, tasks, and Malmo schemas without Java dependencies
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class MinecraftConstants:
    """Minecraft constants and metadata"""
    
    # Basic items that can be crafted/obtained
    ALL_CRAFT_SMELT_ITEMS = [
        'cobblestone', 'stone', 'coal', 'iron_ore', 'diamond',
        'wooden_slab', 'stone_stairs', 'painting', 'iron_ingot',
        'wooden_planks', 'stick', 'wooden_pickaxe', 'stone_pickaxe',
        'iron_pickaxe', 'diamond_pickaxe', 'furnace', 'crafting_table'
    ]
    
    # Biomes
    BIOMES = ['Plains', 'Forest', 'Mountain', 'Desert', 'Swamp']
    
    # Basic recipes (simplified)
    RECIPES = {
        'wooden_planks': {'wood': 1},
        'stick': {'wooden_planks': 2},
        'wooden_pickaxe': {'wooden_planks': 3, 'stick': 2},
        'stone_pickaxe': {'cobblestone': 3, 'stick': 2},
        'iron_pickaxe': {'iron_ingot': 3, 'stick': 2},
        'diamond_pickaxe': {'diamond': 3, 'stick': 2},
        'furnace': {'cobblestone': 8},
        'crafting_table': {'wooden_planks': 4},
        'wooden_slab': {'wooden_planks': 3},
        'stone_stairs': {'cobblestone': 6},
    }
    
    # Smelting recipes
    SMELTING = {
        'iron_ingot': {'iron_ore': 1},
        'stone': {'cobblestone': 1},
    }


class TaskRegistry:
    """Task definitions without Java dependencies"""
    
    def __init__(self):
        self.tasks = {
            'obtain_wooden_slab': {
                'type': 'craft',
                'target': 'wooden_slab',
                'description': 'Obtain a wooden slab by crafting',
                'preconditions': {'wooden_planks': 3}
            },
            'obtain_stone_stairs': {
                'type': 'craft', 
                'target': 'stone_stairs',
                'description': 'Obtain stone stairs by crafting',
                'preconditions': {'cobblestone': 6}
            },
            'obtain_painting': {
                'type': 'craft',
                'target': 'painting',
                'description': 'Obtain a painting by crafting',
                'preconditions': {'stick': 8, 'wool': 1}
            },
            'mine_cobblestone': {
                'type': 'mine',
                'target': 'cobblestone',
                'description': 'Mine cobblestone from stone',
                'preconditions': {'wooden_pickaxe': 1}
            },
            'mine_iron_ore': {
                'type': 'mine',
                'target': 'iron_ore', 
                'description': 'Mine iron ore',
                'preconditions': {'stone_pickaxe': 1}
            },
            'mine_diamond': {
                'type': 'mine',
                'target': 'diamond',
                'description': 'Mine diamond',
                'preconditions': {'iron_pickaxe': 1}
            }
        }
    
    def get_task(self, task_name: str) -> Optional[Dict]:
        """Get task definition"""
        return self.tasks.get(task_name)
    
    def list_tasks(self) -> List[str]:
        """List all available tasks"""
        return list(self.tasks.keys())


class DataManager:
    """Manages data without external dependencies"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def load_goal_mapping(self) -> Dict:
        """Load goal mapping configuration"""
        goal_mapping_file = self.data_dir / "goal_mapping.json"
        if goal_mapping_file.exists():
            with open(goal_mapping_file, 'r') as f:
                return json.load(f)
        else:
            # Return default mapping
            return {
                "mineclip": {
                    "obtain_wooden_slab": "a wooden slab",
                    "obtain_stone_stairs": "stone stairs",
                    "obtain_painting": "a painting"
                },
                "clip": {
                    "obtain_wooden_slab": "wooden slab",
                    "obtain_stone_stairs": "stone stairs", 
                    "obtain_painting": "painting"
                },
                "horizon": {
                    "obtain_wooden_slab": "wooden_slab",
                    "obtain_stone_stairs": "stone_stairs",
                    "obtain_painting": "painting"
                }
            }
    
    def load_task_info(self) -> Dict:
        """Load task information"""
        task_info_file = self.data_dir / "task_info.json"
        if task_info_file.exists():
            with open(task_info_file, 'r') as f:
                return json.load(f)
        else:
            # Return default task info
            return {
                "obtain_wooden_slab": {
                    "group": "crafting",
                    "difficulty": "easy",
                    "preconditions": {"wooden_planks": 3}
                },
                "obtain_stone_stairs": {
                    "group": "crafting", 
                    "difficulty": "medium",
                    "preconditions": {"cobblestone": 6}
                },
                "obtain_painting": {
                    "group": "crafting",
                    "difficulty": "medium", 
                    "preconditions": {"stick": 8, "wool": 1}
                }
            }
    
    def save_data(self, filename: str, data: Dict):
        """Save data to file"""
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class SimulationState:
    """Simplified simulation state management"""
    
    def __init__(self):
        self.inventory = {}
        self.position = [0, 0, 0]
        self.health = 20
        self.hunger = 20
        self.current_biome = "Plains"
        
    def update_inventory(self, item: str, quantity: int):
        """Update inventory item quantity"""
        if item in self.inventory:
            self.inventory[item] += quantity
        else:
            self.inventory[item] = quantity
        
        # Ensure non-negative quantities
        self.inventory[item] = max(0, self.inventory[item])
        
        # Remove items with zero quantity
        if self.inventory[item] == 0:
            del self.inventory[item]
    
    def get_inventory(self) -> Dict[str, int]:
        """Get current inventory"""
        return self.inventory.copy()
    
    def can_craft(self, item: str) -> bool:
        """Check if item can be crafted with current inventory"""
        if item not in MinecraftConstants.RECIPES:
            return False
        
        recipe = MinecraftConstants.RECIPES[item]
        for ingredient, required_amount in recipe.items():
            if self.inventory.get(ingredient, 0) < required_amount:
                return False
        
        return True
    
    def craft_item(self, item: str) -> bool:
        """Craft an item if possible"""
        if not self.can_craft(item):
            return False
        
        recipe = MinecraftConstants.RECIPES[item]
        
        # Consume ingredients
        for ingredient, required_amount in recipe.items():
            self.update_inventory(ingredient, -required_amount)
        
        # Add crafted item
        self.update_inventory(item, 1)
        
        return True


# Global instances
mc = MinecraftConstants()
task_registry = TaskRegistry()
data_manager = DataManager()
