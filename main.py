"""
MC-Planner Main Entry Point
Migrated to gymnasium environment with optimized configuration
"""

import os
import sys
import json
import warnings
import time
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
from wandb_integration import WandBLogger, WandBIntegratedBenchmark
from benchmark_metrics import BenchmarkMetrics

warnings.filterwarnings('ignore')


class Evaluator:
    """Main evaluator class for MC-Planner experiments"""
    
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize WandB logging
        self.wandb_logger = WandBLogger(
            project_name=cfg.get('wandb', {}).get('project', 'diffu-moe-vlm-minecraft'),
            experiment_name=cfg.get('wandb', {}).get('experiment_name', None),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.get('wandb', {}).get('tags', ['minecraft', 'vlm', 'planning']),
            notes=cfg.get('wandb', {}).get('notes', None),
            enabled=cfg.get('wandb', {}).get('enabled', True)
        )
        
        # Initialize benchmark metrics
        self.benchmark_metrics = BenchmarkMetrics()
        
        # Initialize integrated benchmark
        self.integrated_benchmark = WandBIntegratedBenchmark(
            self.benchmark_metrics,
            self.wandb_logger
        )
        
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
        print(f"[Progress] WandB logging enabled: {self.wandb_logger.enabled}")
        
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
        
        # Start episode tracking
        episode_id = f"{task}_{int(time.time())}"
        self.integrated_benchmark.start_episode(episode_id, task)
        
        # Initial planning
        planning_start = time.time()
        plan = self.planner.initial_planning(
            group=task_info.get("group", "crafting"),
            task_question=f"How to {task.replace('_', ' ')}?"
        )
        planning_time = time.time() - planning_start
        
        goal_list = self.planner.generate_goal_list(plan)
        print(f"[Planning] Initial plan: {plan}")
        print(f"[Planning] Goal list: {goal_list}")
        
        # Log initial planning
        self.integrated_benchmark.log_planning(
            plan=plan,
            goal_list=goal_list,
            planning_time=planning_time,
            is_replanning=False
        )
        
        # Main evaluation loop
        success = False
        total_reward = 0.0
        episode_start_time = time.time()
        
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
                
                # Log goal selection
                self.wandb_logger.log_planning_metrics(
                    plan=plan,
                    goal_list=goal_list,
                    selected_goal=selected_goal,
                    planning_time=0.0,
                    is_replanning=False
                )
                
                # Execute action (simplified)
                action = self.generate_action(selected_goal, current_inventory)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                
                # Log step metrics
                self.integrated_benchmark.log_step(
                    obs=obs,
                    action=action,
                    reward=reward,
                    info=info,
                    planning_time=0.0
                )
                
                # Log media (if enabled in config)
                if self.cfg.get('wandb', {}).get('log_images', False) and step % 10 == 0:
                    if "rgb" in obs:
                        self.wandb_logger.log_media(
                            step=step,
                            rgb_obs=obs["rgb"],
                            caption=f"Step {step} - Goal: {selected_goal}"
                        )
                
                if terminated or truncated:
                    break
            else:
                # Replan if no goals available
                planning_start = time.time()
                plan = self.replan_task(current_inventory, f"How to {task.replace('_', ' ')}?")
                planning_time = time.time() - planning_start
                
                goal_list = self.planner.generate_goal_list(plan)
                print(f"[Replan] New plan: {plan}")
                
                # Log replanning
                self.integrated_benchmark.log_planning(
                    plan=plan,
                    goal_list=goal_list,
                    planning_time=planning_time,
                    is_replanning=True
                )
        
        # Calculate completion time
        completion_time = time.time() - episode_start_time
        
        # End episode tracking
        episode_metrics = self.integrated_benchmark.end_episode(
            success=success,
            final_inventory=current_inventory,
            task_id=task
        )
        
        # Complete task evaluation
        task_result = self.integrated_benchmark.complete_task(
            task_id=task,
            success=success,
            completion_time=completion_time,
            total_steps=self.current_step,
            total_reward=total_reward,
            final_inventory=current_inventory,
            planning_iterations=episode_metrics.replanning_count,
            goal_changes=0  # Could be tracked separately
        )
        
        result = {
            'task': task,
            'success': success,
            'steps': self.current_step,
            'total_reward': total_reward,
            'completion_time': completion_time,
            'final_inventory': current_inventory,
            'efficiency_score': task_result.efficiency_score
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
        
        print(f"[Benchmark] Starting evaluation of {len(self.task_list)} tasks")
        
        for i, task in enumerate(self.task_list):
            try:
                print(f"[Benchmark] Evaluating task {i+1}/{len(self.task_list)}: {task}")
                result = self.single_task_evaluate(task)
                results.append(result)
                
                # Log task completion rate
                successful_tasks = sum(1 for r in results if r.get('success', False))
                self.wandb_logger.wandb.log({
                    "benchmark/current_task_index": i + 1,
                    "benchmark/current_success_rate": (successful_tasks / len(results)) * 100,
                    "benchmark/tasks_completed": len(results),
                    "benchmark/tasks_remaining": len(self.task_list) - len(results)
                })
                
            except Exception as e:
                print(f"[Error] Failed to evaluate task {task}: {e}")
                results.append({
                    'task': task,
                    'success': False,
                    'error': str(e),
                    'steps': 0,
                    'total_reward': 0.0,
                    'completion_time': 0.0,
                    'final_inventory': {}
                })
                
                # Log error to WandB
                self.wandb_logger.wandb.log({
                    "errors/task_failures": self.wandb_logger.wandb.run.summary.get("errors/task_failures", 0) + 1,
                    "errors/latest_error": str(e)
                })
        
        # Finish benchmark and log final results
        benchmark_suite = self.integrated_benchmark.finish_benchmark()
        
        # Print summary
        successful_tasks = sum(1 for r in results if r.get('success', False))
        total_steps = sum(r.get('steps', 0) for r in results)
        total_reward = sum(r.get('total_reward', 0.0) for r in results)
        total_time = sum(r.get('completion_time', 0.0) for r in results)
        
        print(f"\n[Summary] Benchmark completed!")
        print(f"[Summary] Tasks completed: {successful_tasks}/{len(results)}")
        print(f"[Summary] Overall success rate: {(successful_tasks/len(results)*100):.2f}%")
        print(f"[Summary] Total steps: {total_steps}")
        print(f"[Summary] Total reward: {total_reward:.2f}")
        print(f"[Summary] Total time: {total_time:.2f}s")
        print(f"[Summary] Average steps per task: {total_steps/len(results):.2f}")
        print(f"[Summary] Average reward per task: {total_reward/len(results):.2f}")
        
        return results


@hydra.main(config_path="configs", config_name="defaults", version_base=None)
def main(cfg: OmegaConf) -> None:
    """Main entry point"""
    print("Starting MC-Planner Migration with WandB Integration...")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Initialize evaluator
    evaluator = Evaluator(cfg)
    
    try:
        # Run evaluation
        if cfg.get('single_task'):
            # Run single task
            task = cfg.get('task_name', evaluator.task_list[0])
            result = evaluator.single_task_evaluate(task)
            
            # Save single task result
            output_file = Path(cfg.get('output_dir', '.')) / f'result_{task}.json'
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Single task result saved to {output_file}")
            
        else:
            # Run all tasks
            results = evaluator.run_all_tasks()
            
            # Save results
            output_file = Path(cfg.get('output_dir', '.')) / 'results.json'
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_file}")
            
            # Log results file as WandB artifact
            evaluator.wandb_logger.log_artifact(
                str(output_file),
                "final_results",
                "results"
            )
    
    except Exception as e:
        print(f"[Error] Evaluation failed: {e}")
        
        # Log error to WandB
        if hasattr(evaluator, 'wandb_logger') and evaluator.wandb_logger.enabled:
            evaluator.wandb_logger.wandb.log({
                "errors/fatal_error": str(e),
                "errors/evaluation_failed": True
            })
        
        raise e
    
    finally:
        # Ensure WandB is properly closed
        if hasattr(evaluator, 'wandb_logger') and evaluator.wandb_logger.enabled:
            evaluator.wandb_logger.finish()
            print("[WandB] Logging session finished")


if __name__ == '__main__':
    main()
