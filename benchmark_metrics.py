"""
MineDojo-style Benchmarking Metrics for Diffu_MoE_VLM
Implements comprehensive evaluation metrics based on MineDojo's benchmarking suite
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class TaskResult:
    """Single task evaluation result"""
    task_id: str
    task_type: str  # "programmatic", "creative", "playthrough"
    success: bool
    completion_time: float
    total_steps: int
    reward: float
    final_inventory: Dict[str, int]
    trajectory_length: int
    completion_percentage: float  # 0-100
    efficiency_score: float  # reward per step
    planning_iterations: int
    goal_changes: int
    error_message: Optional[str] = None


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    episode_id: str
    start_time: float
    end_time: float
    total_steps: int
    total_reward: float
    success: bool
    
    # MineDojo-style metrics
    inventory_diversity: int  # Number of unique items obtained
    crafting_events: int  # Number of crafting actions
    mining_events: int  # Number of mining actions
    exploration_area: float  # Area explored (in blocks²)
    
    # Planning metrics
    planning_time: float
    replanning_count: int
    goal_completion_rate: float
    
    # Efficiency metrics
    actions_per_second: float
    reward_per_step: float
    time_to_first_success: Optional[float]


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: float
    end_time: float
    total_tasks: int
    completed_tasks: int
    successful_tasks: int
    
    # Task category breakdown
    programmatic_results: List[TaskResult]
    creative_results: List[TaskResult]
    playthrough_results: List[TaskResult]
    
    # Aggregate metrics
    overall_success_rate: float
    average_completion_time: float
    average_steps_per_task: float
    average_reward: float
    
    # MineDojo-specific metrics
    tech_tree_progression: Dict[str, bool]  # Which tools/items were successfully crafted
    survival_metrics: Dict[str, float]  # Health, hunger, survival time
    combat_metrics: Dict[str, int]  # Monsters defeated, damage taken/dealt


class MinecraftTaskCategories:
    """Task categorization based on MineDojo benchmark suite"""
    
    SURVIVAL_TASKS = [
        "survive_1_day", "survive_3_days", "survive_7_days",
        "survive_with_food", "survive_with_sword"
    ]
    
    HARVEST_TASKS = [
        "harvest_milk", "harvest_wool", "obtain_cobblestone",
        "obtain_iron_ore", "obtain_diamond", "obtain_wooden_planks"
    ]
    
    TECH_TREE_TASKS = [
        "craft_wooden_pickaxe", "craft_stone_pickaxe", "craft_iron_pickaxe",
        "craft_diamond_pickaxe", "craft_wooden_sword", "craft_iron_sword"
    ]
    
    COMBAT_TASKS = [
        "combat_zombie", "combat_skeleton", "combat_spider",
        "hunt_pig", "hunt_cow", "hunt_chicken"
    ]
    
    CREATIVE_TASKS = [
        "build_house", "build_castle", "build_bridge",
        "create_art", "build_farm", "build_tower"
    ]
    
    @classmethod
    def get_task_category(cls, task_id: str) -> str:
        """Determine task category from task ID"""
        if any(task in task_id for task in cls.SURVIVAL_TASKS):
            return "survival"
        elif any(task in task_id for task in cls.HARVEST_TASKS):
            return "harvest"
        elif any(task in task_id for task in cls.TECH_TREE_TASKS):
            return "tech_tree"
        elif any(task in task_id for task in cls.COMBAT_TASKS):
            return "combat"
        elif any(task in task_id for task in cls.CREATIVE_TASKS):
            return "creative"
        elif "creative:" in task_id:
            return "creative"
        elif "playthrough" in task_id:
            return "playthrough"
        else:
            return "programmatic"


class BenchmarkMetrics:
    """
    Comprehensive benchmarking metrics system following MineDojo's evaluation framework
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metric tracking
        self.current_episode_metrics = None
        self.task_results = []
        self.episode_history = []
        
        # MineDojo-style tracking
        self.tech_tree_items = {
            "wooden_pickaxe": False, "stone_pickaxe": False, "iron_pickaxe": False,
            "diamond_pickaxe": False, "wooden_sword": False, "iron_sword": False,
            "diamond_sword": False, "bow": False, "crossbow": False
        }
        
        self.survival_stats = {
            "max_health": 20.0,
            "current_health": 20.0,
            "hunger_level": 20.0,
            "days_survived": 0.0
        }
        
        self.combat_stats = {
            "monsters_defeated": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "deaths": 0
        }
        
    def start_episode(self, episode_id: str, task_id: str) -> None:
        """Start tracking a new episode"""
        self.current_episode_metrics = EpisodeMetrics(
            episode_id=episode_id,
            start_time=time.time(),
            end_time=0.0,
            total_steps=0,
            total_reward=0.0,
            success=False,
            inventory_diversity=0,
            crafting_events=0,
            mining_events=0,
            exploration_area=0.0,
            planning_time=0.0,
            replanning_count=0,
            goal_completion_rate=0.0,
            actions_per_second=0.0,
            reward_per_step=0.0,
            time_to_first_success=None
        )
        
        print(f"[Benchmark] Started episode {episode_id} for task {task_id}")
    
    def update_step_metrics(self, 
                          obs: Dict,
                          action: Dict,
                          reward: float,
                          info: Dict) -> None:
        """Update metrics for each step"""
        if not self.current_episode_metrics:
            return
            
        self.current_episode_metrics.total_steps += 1
        self.current_episode_metrics.total_reward += reward
        
        # Update inventory metrics
        inventory = info.get('inventory', {})
        self.current_episode_metrics.inventory_diversity = len(inventory)
        
        # Track crafting and mining events
        if action.get('craft', 0) > 0:
            self.current_episode_metrics.crafting_events += 1
            
        if action.get('attack', 0) > 0:
            self.current_episode_metrics.mining_events += 1
        
        # Update tech tree progression
        for item in inventory:
            if item in self.tech_tree_items:
                self.tech_tree_items[item] = True
        
        # Update survival stats
        self.survival_stats.update({
            "current_health": info.get('life_stats', {}).get('health', 20.0),
            "hunger_level": info.get('life_stats', {}).get('food', 20.0)
        })
        
        # Update exploration area (simplified)
        pos = info.get('location_stats', {}).get('pos', [0, 0, 0])
        if len(pos) >= 2:
            # Simple area calculation based on max distance from origin
            area = max(abs(pos[0]), abs(pos[2])) ** 2
            self.current_episode_metrics.exploration_area = max(
                self.current_episode_metrics.exploration_area, area
            )
    
    def update_planning_metrics(self, 
                              planning_time: float,
                              is_replanning: bool = False) -> None:
        """Update planning-related metrics"""
        if not self.current_episode_metrics:
            return
            
        self.current_episode_metrics.planning_time += planning_time
        
        if is_replanning:
            self.current_episode_metrics.replanning_count += 1
    
    def end_episode(self, success: bool, final_inventory: Dict) -> EpisodeMetrics:
        """End episode tracking and compute final metrics"""
        if not self.current_episode_metrics:
            raise ValueError("No active episode to end")
        
        self.current_episode_metrics.end_time = time.time()
        self.current_episode_metrics.success = success
        
        # Compute derived metrics
        episode_duration = (self.current_episode_metrics.end_time - 
                          self.current_episode_metrics.start_time)
        
        if episode_duration > 0:
            self.current_episode_metrics.actions_per_second = (
                self.current_episode_metrics.total_steps / episode_duration
            )
        
        if self.current_episode_metrics.total_steps > 0:
            self.current_episode_metrics.reward_per_step = (
                self.current_episode_metrics.total_reward / 
                self.current_episode_metrics.total_steps
            )
        
        # Store episode
        episode_metrics = self.current_episode_metrics
        self.episode_history.append(episode_metrics)
        self.current_episode_metrics = None
        
        return episode_metrics
    
    def evaluate_task(self, 
                     task_id: str,
                     success: bool,
                     completion_time: float,
                     total_steps: int,
                     total_reward: float,
                     final_inventory: Dict,
                     planning_iterations: int = 0,
                     goal_changes: int = 0,
                     error_message: Optional[str] = None) -> TaskResult:
        """Evaluate a complete task"""
        
        task_category = MinecraftTaskCategories.get_task_category(task_id)
        task_type = "programmatic" if task_category != "creative" else "creative"
        if "playthrough" in task_id:
            task_type = "playthrough"
        
        # Calculate completion percentage
        completion_percentage = 100.0 if success else 0.0
        
        # Calculate efficiency score
        efficiency_score = total_reward / max(total_steps, 1)
        
        result = TaskResult(
            task_id=task_id,
            task_type=task_type,
            success=success,
            completion_time=completion_time,
            total_steps=total_steps,
            reward=total_reward,
            final_inventory=final_inventory,
            trajectory_length=total_steps,
            completion_percentage=completion_percentage,
            efficiency_score=efficiency_score,
            planning_iterations=planning_iterations,
            goal_changes=goal_changes,
            error_message=error_message
        )
        
        self.task_results.append(result)
        return result
    
    def compute_benchmark_summary(self) -> BenchmarkSuite:
        """Compute comprehensive benchmark summary"""
        if not self.task_results:
            raise ValueError("No task results to summarize")
        
        # Categorize results
        programmatic_results = [r for r in self.task_results if r.task_type == "programmatic"]
        creative_results = [r for r in self.task_results if r.task_type == "creative"]
        playthrough_results = [r for r in self.task_results if r.task_type == "playthrough"]
        
        # Compute aggregate metrics
        successful_tasks = sum(1 for r in self.task_results if r.success)
        total_tasks = len(self.task_results)
        
        overall_success_rate = successful_tasks / max(total_tasks, 1) * 100
        average_completion_time = np.mean([r.completion_time for r in self.task_results])
        average_steps_per_task = np.mean([r.total_steps for r in self.task_results])
        average_reward = np.mean([r.reward for r in self.task_results])
        
        return BenchmarkSuite(
            suite_name="MineDojo_Benchmark",
            start_time=min(episode.start_time for episode in self.episode_history) if self.episode_history else time.time(),
            end_time=max(episode.end_time for episode in self.episode_history) if self.episode_history else time.time(),
            total_tasks=total_tasks,
            completed_tasks=total_tasks,
            successful_tasks=successful_tasks,
            programmatic_results=programmatic_results,
            creative_results=creative_results,
            playthrough_results=playthrough_results,
            overall_success_rate=overall_success_rate,
            average_completion_time=average_completion_time,
            average_steps_per_task=average_steps_per_task,
            average_reward=average_reward,
            tech_tree_progression=self.tech_tree_items.copy(),
            survival_metrics=self.survival_stats.copy(),
            combat_metrics=self.combat_stats.copy()
        )
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save benchmark results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data for serialization
        data = {
            "benchmark_summary": asdict(self.compute_benchmark_summary()),
            "task_results": [asdict(result) for result in self.task_results],
            "episode_history": [asdict(episode) for episode in self.episode_history],
            "tech_tree_progression": self.tech_tree_items,
            "survival_metrics": self.survival_stats,
            "combat_metrics": self.combat_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"[Benchmark] Results saved to {filepath}")
        return filepath
    
    def print_summary(self) -> None:
        """Print benchmark summary to console"""
        summary = self.compute_benchmark_summary()
        
        print("\n" + "="*60)
        print("MINEDOJO-STYLE BENCHMARK RESULTS")
        print("="*60)
        print(f"Suite: {summary.suite_name}")
        print(f"Total Tasks: {summary.total_tasks}")
        print(f"Successful Tasks: {summary.successful_tasks}")
        print(f"Overall Success Rate: {summary.overall_success_rate:.2f}%")
        print(f"Average Completion Time: {summary.average_completion_time:.2f}s")
        print(f"Average Steps per Task: {summary.average_steps_per_task:.1f}")
        print(f"Average Reward: {summary.average_reward:.3f}")
        
        print("\nTask Category Breakdown:")
        print(f"  Programmatic: {len(summary.programmatic_results)} tasks")
        print(f"  Creative: {len(summary.creative_results)} tasks")
        print(f"  Playthrough: {len(summary.playthrough_results)} tasks")
        
        print("\nTech Tree Progression:")
        for item, obtained in summary.tech_tree_progression.items():
            status = "✓" if obtained else "✗"
            print(f"  {status} {item}")
        
        print("\nSurvival Metrics:")
        for metric, value in summary.survival_metrics.items():
            print(f"  {metric}: {value}")
        
        print("\nCombat Metrics:")
        for metric, value in summary.combat_metrics.items():
            print(f"  {metric}: {value}")
        
        print("="*60)
