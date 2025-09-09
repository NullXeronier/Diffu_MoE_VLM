"""
MC-Planner Main Entry Point
Migrated to gymnasium environment with optimized configuration
"""

import os
import sys
import json
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from omegaconf import OmegaConf
import hydra
from hydra.utils import get_original_cwd, to_absolute_path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.gymnasium_env import MineDojoEnv
from src.minedojo_core import mc, task_registry, data_manager
from planner import Planner
from selector import Selector
from controller import MineAgent, MineAgentWrapper

warnings.filterwarnings('ignore')


class Evaluator:
    """Main evaluator class for MC-Planner experiments"""
    
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize environment
        self.env = MineDojoEnv(
            name=cfg.eval.env_name,
            img_size=(cfg.simulator.resolution[0], cfg.simulator.resolution[1]),
            rgb_only=False,
        )
        
        # Load task list
        self.task_list = list(task_registry.list_tasks())
        
        # Initialize components
        self.planner = Planner()
        self.selector = Selector()
        
        # Load goal mappings and task info
        self.goal_mapping_cfg = data_manager.load_goal_mapping()
        self.task_info = data_manager.load_task_info()
        
        # Configuration
        self.goal_model_freq = cfg.goal_model.freq
        self.goal_list_size = cfg.goal_model.queue_size
        self.record_frames = cfg.record.frames
        
        print(f"[Progress] Initialized evaluator on {self.device}")
        print(f"[Progress] Available tasks: {len(self.task_list)}")
        
    def reset(self, task: str):
        """Reset environment for new task"""
        obs, info = self.env.reset()
        self.current_task = task
        self.current_step = 0
        self.max_steps = self.cfg.eval.max_steps
        
        # Reset planner
        self.planner.reset()
        
        print(f"[Reset] Starting task: {task}")
        return obs, info
    
    def load_task_info(self, task: str) -> Dict:
        """Load task information"""
        return self.task_info.get(task, {})
    
    def check_inventory(self, inventory: Dict, items: Dict) -> bool:
        """Check if inventory contains required items"""
        for item, required_amount in items.items():
            if inventory.get(item, 0) < required_amount:
                return False
        return True
    
    def check_precondition(self, inventory: Dict, precondition: Dict) -> bool:
        """Check if preconditions are met"""
        return self.check_inventory(inventory, precondition)
    
    def check_done(self, inventory: Dict, task_obj: str) -> bool:
        """Check if task is completed"""
        return inventory.get(task_obj, 0) > 0
    
    def update_goal(self, inventory: Dict) -> List[str]:
        """Update goal list based on current state"""
        # Simple goal selection based on what can be crafted
        possible_goals = []
        
        for item in mc.ALL_CRAFT_SMELT_ITEMS:
            if item in mc.RECIPES:
                recipe = mc.RECIPES[item]
                if self.check_inventory(inventory, recipe):
                    possible_goals.append(f"obtain_{item}")
        
        return possible_goals[:self.goal_list_size]
    
    def replan_task(self, inventory: Dict, task_question: str) -> str:
        """Replan task based on current state"""
        inventory_desc = self.generate_inventory_description(inventory)
        return self.planner.replan(task_question, inventory_desc)
    
    def generate_inventory_description(self, inventory: Dict) -> str:
        """Generate natural language description of inventory"""
        if not inventory:
            return "The inventory is empty."
        
        items = []
        for item, count in inventory.items():
            if count > 1:
                items.append(f"{count} {item}s")
            else:
                items.append(f"{count} {item}")
        
        return f"The inventory contains: {', '.join(items)}."
    
    def single_task_evaluate(self, task: str = None):
        """Evaluate a single task"""
        if task is None:
            task = self.task_list[0]  # Default to first task
        
        # Reset environment
        obs, info = self.reset(task)
        
        # Load task information
        task_info = self.load_task_info(task)
        task_obj = task.replace("obtain_", "").replace("mine_", "")
        
        print(f"[Evaluation] Starting task: {task}")
        print(f"[Evaluation] Target object: {task_obj}")
        
        # Initial planning
        plan = self.planner.initial_planning(
            group=task_info.get("group", "crafting"),
            task_question=f"How to {task.replace('_', ' ')}?"
        )
        
        goal_list = self.planner.generate_goal_list(plan)
        print(f"[Planning] Initial plan: {plan}")
        print(f"[Planning] Goal list: {goal_list}")
        
        # Main evaluation loop
        success = False
        for step in range(self.max_steps):
            self.current_step = step
            
            # Get current inventory from environment
            current_inventory = info.get('inventory', {})
            
            # Check if task is completed
            if self.check_done(current_inventory, task_obj):
                success = True
                print(f"[Success] Task completed at step {step}")
                break
            
            # Update goals based on current state
            if step % self.goal_model_freq == 0:
                goal_list = self.update_goal(current_inventory)
            
            # Select next goal
            if goal_list:
                selected_goal = self.selector.horizon_select(goal_list)
                print(f"[Step {step}] Selected goal: {selected_goal}")
                
                # Execute action (simplified)
                action = self.generate_action(selected_goal, current_inventory)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated or truncated:
                    break
            else:
                # Replan if no goals available
                plan = self.replan_task(current_inventory, f"How to {task.replace('_', ' ')}?")
                goal_list = self.planner.generate_goal_list(plan)
                print(f"[Replan] New plan: {plan}")
        
        result = {
            'task': task,
            'success': success,
            'steps': self.current_step,
            'final_inventory': info.get('inventory', {})
        }
        
        print(f"[Result] {result}")
        return result
    
    def generate_action(self, goal: str, inventory: Dict) -> Dict:
        """Generate action for given goal (simplified)"""
        # Simplified action generation
        action = {
            'movement': 0,  # forward
            'camera': np.array([0.0, 0.0]),
            'use': 0,
            'attack': 0,
            'jump': 0
        }
        
        # Basic action logic based on goal
        if "mine" in goal:
            action['attack'] = 1
        elif "craft" in goal:
            action['use'] = 1
        
        return action
    
    def run_all_tasks(self):
        """Run evaluation on all tasks"""
        results = []
        
        for task in self.task_list:
            try:
                result = self.single_task_evaluate(task)
                results.append(result)
            except Exception as e:
                print(f"[Error] Failed to evaluate task {task}: {e}")
                results.append({
                    'task': task,
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        successful_tasks = sum(1 for r in results if r.get('success', False))
        print(f"\n[Summary] Completed {successful_tasks}/{len(results)} tasks successfully")
        
        return results


@hydra.main(config_path="configs", config_name="defaults", version_base=None)
def main(cfg: OmegaConf) -> None:
    """Main entry point"""
    print("Starting MC-Planner Migration...")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Initialize evaluator
    evaluator = Evaluator(cfg)
    
    # Run evaluation
    if cfg.get('single_task'):
        # Run single task
        task = cfg.get('task_name', evaluator.task_list[0])
        result = evaluator.single_task_evaluate(task)
    else:
        # Run all tasks
        results = evaluator.run_all_tasks()
        
        # Save results
        output_file = Path(cfg.get('output_dir', '.')) / 'results.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
