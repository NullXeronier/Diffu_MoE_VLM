"""
MineDojo Task Configuration for Diffu_MoE_VLM
Maps MineDojo benchmark tasks to the current project structure
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path


class TaskRegistry:
    """Registry for MineDojo-style tasks adapted for Diffu_MoE_VLM"""
    
    # Programmatic tasks from MineDojo benchmark suite
    PROGRAMMATIC_TASKS = {
        # Survival tasks
        "survive_1_day": {
            "type": "survival",
            "description": "Survive for 1 Minecraft day (20 minutes)",
            "success_criteria": {"days_survived": 1},
            "max_steps": 24000,  # 20 minutes at 20 TPS
            "difficulty": "easy"
        },
        "survive_3_days": {
            "type": "survival", 
            "description": "Survive for 3 Minecraft days",
            "success_criteria": {"days_survived": 3},
            "max_steps": 72000,
            "difficulty": "medium"
        },
        
        # Harvest tasks
        "obtain_wooden_slab": {
            "type": "harvest",
            "description": "Obtain wooden slab from wooden planks",
            "success_criteria": {"wooden_slab": 1},
            "max_steps": 1000,
            "difficulty": "easy",
            "required_items": ["wooden_planks"]
        },
        "obtain_stone_stairs": {
            "type": "harvest",
            "description": "Craft stone stairs from cobblestone", 
            "success_criteria": {"stone_stairs": 1},
            "max_steps": 2000,
            "difficulty": "medium",
            "required_items": ["cobblestone"]
        },
        "obtain_painting": {
            "type": "harvest",
            "description": "Craft painting from sticks and wool",
            "success_criteria": {"painting": 1},
            "max_steps": 3000,
            "difficulty": "medium",
            "required_items": ["stick", "wool"]
        },
        "harvest_milk": {
            "type": "harvest",
            "description": "Obtain milk from a cow",
            "success_criteria": {"milk_bucket": 1},
            "max_steps": 2000,
            "difficulty": "easy",
            "required_items": ["bucket"]
        },
        "harvest_wool": {
            "type": "harvest",
            "description": "Shear wool from a sheep",
            "success_criteria": {"wool": 1},
            "max_steps": 1500,
            "difficulty": "easy",
            "required_items": ["shears"]
        },
        
        # Mining tasks
        "mine_cobblestone": {
            "type": "mining",
            "description": "Mine cobblestone with pickaxe",
            "success_criteria": {"cobblestone": 8},
            "max_steps": 1500,
            "difficulty": "easy",
            "required_items": ["wooden_pickaxe"]
        },
        "mine_iron_ore": {
            "type": "mining",
            "description": "Mine iron ore with stone pickaxe",
            "success_criteria": {"iron_ore": 3},
            "max_steps": 3000,
            "difficulty": "medium",
            "required_items": ["stone_pickaxe"]
        },
        "mine_diamond": {
            "type": "mining",
            "description": "Mine diamond with iron pickaxe",
            "success_criteria": {"diamond": 1},
            "max_steps": 5000,
            "difficulty": "hard",
            "required_items": ["iron_pickaxe"]
        },
        
        # Tech tree tasks
        "craft_wooden_pickaxe": {
            "type": "tech_tree",
            "description": "Craft wooden pickaxe from planks and sticks",
            "success_criteria": {"wooden_pickaxe": 1},
            "max_steps": 1000,
            "difficulty": "easy",
            "required_items": ["wooden_planks", "stick"]
        },
        "craft_stone_pickaxe": {
            "type": "tech_tree",
            "description": "Craft stone pickaxe from cobblestone and sticks",
            "success_criteria": {"stone_pickaxe": 1},
            "max_steps": 2000,
            "difficulty": "medium",
            "required_items": ["cobblestone", "stick"]
        },
        "craft_iron_pickaxe": {
            "type": "tech_tree",
            "description": "Craft iron pickaxe from iron ingots and sticks",
            "success_criteria": {"iron_pickaxe": 1},
            "max_steps": 4000,
            "difficulty": "hard",
            "required_items": ["iron_ingot", "stick"]
        },
        "craft_diamond_pickaxe": {
            "type": "tech_tree",
            "description": "Craft diamond pickaxe from diamonds and sticks",
            "success_criteria": {"diamond_pickaxe": 1},
            "max_steps": 6000,
            "difficulty": "very_hard",
            "required_items": ["diamond", "stick"]
        },
        
        # Combat tasks
        "combat_zombie": {
            "type": "combat",
            "description": "Defeat a zombie with sword",
            "success_criteria": {"zombies_defeated": 1},
            "max_steps": 2000,
            "difficulty": "medium",
            "required_items": ["wooden_sword"]
        },
        "combat_skeleton": {
            "type": "combat", 
            "description": "Defeat a skeleton with sword and shield",
            "success_criteria": {"skeletons_defeated": 1},
            "max_steps": 2500,
            "difficulty": "hard",
            "required_items": ["iron_sword", "shield"]
        },
        "hunt_pig": {
            "type": "combat",
            "description": "Hunt a pig for food",
            "success_criteria": {"porkchop": 1},
            "max_steps": 1000,
            "difficulty": "easy",
            "required_items": ["wooden_sword"]
        }
    }
    
    # Creative tasks (procedurally generated or manually specified)
    CREATIVE_TASKS = {
        "build_house": {
            "type": "creative",
            "description": "Build a simple house with walls, roof, and door",
            "success_criteria": {"structure_complexity": 50},
            "max_steps": 5000,
            "difficulty": "medium"
        },
        "build_castle": {
            "type": "creative",
            "description": "Build a castle with towers and walls",
            "success_criteria": {"structure_complexity": 200},
            "max_steps": 10000,
            "difficulty": "hard"
        },
        "build_bridge": {
            "type": "creative",
            "description": "Build a bridge across water or ravine",
            "success_criteria": {"bridge_length": 20},
            "max_steps": 3000,
            "difficulty": "medium"
        },
        "create_art": {
            "type": "creative",
            "description": "Create pixel art using colored blocks",
            "success_criteria": {"art_complexity": 30},
            "max_steps": 2000,
            "difficulty": "easy"
        },
        "build_farm": {
            "type": "creative",
            "description": "Build a functioning farm with crops",
            "success_criteria": {"farm_size": 9},  # 3x3 minimum
            "max_steps": 4000,
            "difficulty": "medium"
        }
    }
    
    # Playthrough task (ultimate challenge)
    PLAYTHROUGH_TASK = {
        "defeat_ender_dragon": {
            "type": "playthrough",
            "description": "Defeat the Ender Dragon and obtain the dragon egg",
            "success_criteria": {"dragon_egg": 1},
            "max_steps": 100000,  # Very long task
            "difficulty": "extreme"
        }
    }
    
    @classmethod
    def get_all_tasks(cls) -> Dict[str, Dict]:
        """Get all tasks combined"""
        all_tasks = {}
        all_tasks.update(cls.PROGRAMMATIC_TASKS)
        all_tasks.update(cls.CREATIVE_TASKS)
        all_tasks.update(cls.PLAYTHROUGH_TASK)
        return all_tasks
    
    @classmethod
    def get_tasks_by_type(cls, task_type: str) -> Dict[str, Dict]:
        """Get tasks filtered by type"""
        all_tasks = cls.get_all_tasks()
        return {k: v for k, v in all_tasks.items() if v["type"] == task_type}
    
    @classmethod
    def get_tasks_by_difficulty(cls, difficulty: str) -> Dict[str, Dict]:
        """Get tasks filtered by difficulty"""
        all_tasks = cls.get_all_tasks()
        return {k: v for k, v in all_tasks.items() if v["difficulty"] == difficulty}
    
    @classmethod
    def get_task_info(cls, task_id: str) -> Optional[Dict]:
        """Get information for a specific task"""
        all_tasks = cls.get_all_tasks()
        return all_tasks.get(task_id)
    
    @classmethod
    def list_task_ids(cls) -> List[str]:
        """List all available task IDs"""
        return list(cls.get_all_tasks().keys())
    
    @classmethod
    def save_task_registry(cls, filepath: str):
        """Save task registry to JSON file"""
        all_tasks = cls.get_all_tasks()
        with open(filepath, 'w') as f:
            json.dump(all_tasks, f, indent=2)
        print(f"Task registry saved to {filepath}")
    
    @classmethod
    def create_task_mapping(cls) -> Dict[str, str]:
        """Create mapping from MineDojo task IDs to goal descriptions"""
        all_tasks = cls.get_all_tasks()
        return {task_id: info["description"] for task_id, info in all_tasks.items()}


class MinedojoTaskAdapter:
    """Adapter to integrate MineDojo tasks with the current evaluation system"""
    
    def __init__(self):
        self.task_registry = TaskRegistry()
        self.current_task = None
        self.task_progress = {}
    
    def start_task(self, task_id: str) -> Dict:
        """Start a new task and return its configuration"""
        task_info = self.task_registry.get_task_info(task_id)
        if not task_info:
            raise ValueError(f"Unknown task: {task_id}")
        
        self.current_task = task_id
        self.task_progress = {
            "start_time": 0,
            "steps_taken": 0,
            "items_obtained": {},
            "goals_completed": [],
            "success": False
        }
        
        return task_info
    
    def check_task_completion(self, inventory: Dict[str, int], info: Dict) -> bool:
        """Check if current task is completed based on success criteria"""
        if not self.current_task:
            return False
        
        task_info = self.task_registry.get_task_info(self.current_task)
        if not task_info:
            return False
        
        success_criteria = task_info.get("success_criteria", {})
        
        # Check inventory-based criteria
        for item, required_count in success_criteria.items():
            if item in ["days_survived", "structure_complexity", "bridge_length", "art_complexity", "farm_size"]:
                # These require special handling from environment info
                if info.get(item, 0) < required_count:
                    return False
            elif item.endswith("_defeated"):
                # Combat criteria
                if info.get("combat_stats", {}).get(item, 0) < required_count:
                    return False
            else:
                # Standard inventory check
                if inventory.get(item, 0) < required_count:
                    return False
        
        return True
    
    def get_task_reward(self, inventory: Dict[str, int], info: Dict, step: int) -> float:
        """Calculate reward for current task progress"""
        if not self.current_task:
            return 0.0
        
        task_info = self.task_registry.get_task_info(self.current_task)
        if not task_info:
            return 0.0
        
        # Base reward calculation
        base_reward = 0.0
        success_criteria = task_info.get("success_criteria", {})
        
        # Progress-based reward
        for item, required_count in success_criteria.items():
            if item in inventory:
                progress = min(inventory[item] / required_count, 1.0)
                base_reward += progress * 10.0  # 10 points per criterion
        
        # Completion bonus
        if self.check_task_completion(inventory, info):
            difficulty_multiplier = {
                "easy": 1.0,
                "medium": 1.5,
                "hard": 2.0,
                "very_hard": 3.0,
                "extreme": 5.0
            }.get(task_info.get("difficulty", "medium"), 1.0)
            
            base_reward += 100.0 * difficulty_multiplier
        
        # Efficiency bonus (fewer steps = higher reward)
        max_steps = task_info.get("max_steps", 1000)
        efficiency = max(0, (max_steps - step) / max_steps)
        base_reward += efficiency * 20.0
        
        return base_reward
    
    def get_task_guidance(self, task_id: str) -> List[str]:
        """Get step-by-step guidance for a task"""
        guidance_map = {
            "obtain_wooden_slab": [
                "1. Find trees and mine wood",
                "2. Craft wooden planks from wood", 
                "3. Craft wooden slab from planks"
            ],
            "obtain_stone_stairs": [
                "1. Find stone or mine cobblestone",
                "2. Craft stone stairs from cobblestone"
            ],
            "obtain_painting": [
                "1. Find trees and mine wood",
                "2. Craft wooden planks and then sticks",
                "3. Find sheep and shear wool",
                "4. Craft painting from sticks and wool"
            ],
            "mine_cobblestone": [
                "1. Craft or find a wooden pickaxe",
                "2. Find stone blocks underground",
                "3. Mine stone to get cobblestone"
            ],
            "craft_wooden_pickaxe": [
                "1. Find trees and mine wood",
                "2. Craft wooden planks from wood",
                "3. Craft sticks from planks",
                "4. Craft pickaxe from planks and sticks"
            ]
        }
        
        return guidance_map.get(task_id, ["Complete the task objective"])


# Initialize the task adapter for use in main evaluation
def create_task_adapter() -> MinedojoTaskAdapter:
    """Create and return a configured task adapter"""
    return MinedojoTaskAdapter()


# Export task information for external use
def export_task_info(output_dir: str = "./data"):
    """Export task information to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save complete registry
    TaskRegistry.save_task_registry(str(output_path / "task_registry.json"))
    
    # Save task mapping
    task_mapping = TaskRegistry.create_task_mapping()
    with open(output_path / "task_mapping.json", 'w') as f:
        json.dump(task_mapping, f, indent=2)
    
    # Save task lists by category
    categories = ["survival", "harvest", "mining", "tech_tree", "combat", "creative", "playthrough"]
    for category in categories:
        tasks = TaskRegistry.get_tasks_by_type(category)
        with open(output_path / f"tasks_{category}.json", 'w') as f:
            json.dump(tasks, f, indent=2)
    
    print(f"Task information exported to {output_path}")


if __name__ == "__main__":
    # Export task information when run directly
    export_task_info()
