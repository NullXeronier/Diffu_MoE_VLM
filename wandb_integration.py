"""
Weights & Biases Integration for MineDojo-style Benchmarking
Provides comprehensive logging and visualization for Diffu_MoE_VLM experiments
"""

import os
import time
import wandb
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from benchmark_metrics import BenchmarkMetrics, TaskResult, EpisodeMetrics, BenchmarkSuite


class WandBLogger:
    """
    Weights & Biases logger for MineDojo-style benchmarking
    """
    
    def __init__(self, 
                 project_name: str = "diffu-moe-vlm-minecraft",
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 enabled: bool = True):
        
        self.enabled = enabled
        self.project_name = project_name
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        
        if not self.enabled:
            print("[WandB] Logging disabled")
            return
        
        # Initialize WandB
        try:
            wandb.init(
                project=project_name,
                name=self.experiment_name,
                config=config or {},
                tags=tags or [],
                notes=notes
            )
            
            # Define custom metrics
            self._define_custom_metrics()
            print(f"[WandB] Initialized project: {project_name}, experiment: {self.experiment_name}")
            
        except Exception as e:
            print(f"[WandB] Failed to initialize: {e}")
            self.enabled = False
    
    def _define_custom_metrics(self):
        """Define custom metrics for MineDojo benchmarking"""
        if not self.enabled:
            return
            
        # Define step-based metrics
        wandb.define_metric("episode/step")
        wandb.define_metric("episode/*", step_metric="episode/step")
        
        # Define task-based metrics
        wandb.define_metric("task/index")
        wandb.define_metric("task/*", step_metric="task/index")
        
        # Define benchmark suite metrics
        wandb.define_metric("benchmark/task_count")
        wandb.define_metric("benchmark/*", step_metric="benchmark/task_count")
    
    def log_config(self, config: Dict):
        """Log experiment configuration"""
        if not self.enabled:
            return
            
        wandb.config.update(config)
    
    def log_step_metrics(self, 
                        step: int,
                        reward: float,
                        action: Dict,
                        inventory: Dict,
                        obs: Dict,
                        info: Dict,
                        planning_time: Optional[float] = None):
        """Log metrics for each environment step"""
        if not self.enabled:
            return
        
        metrics = {
            "episode/step": step,
            "episode/reward": reward,
            "episode/cumulative_reward": wandb.run.summary.get("episode/cumulative_reward", 0) + reward,
            "episode/inventory_size": len(inventory),
            "episode/inventory_diversity": len(set(inventory.keys())),
        }
        
        # Log action distribution
        for action_name, action_value in action.items():
            if isinstance(action_value, (int, float)):
                metrics[f"action/{action_name}"] = action_value
        
        # Log inventory details
        for item, count in inventory.items():
            metrics[f"inventory/{item}"] = count
        
        # Log observation statistics
        if "rgb" in obs:
            rgb = obs["rgb"]
            if isinstance(rgb, np.ndarray):
                metrics.update({
                    "obs/rgb_mean": np.mean(rgb),
                    "obs/rgb_std": np.std(rgb),
                })
        
        # Log life stats
        life_stats = info.get("life_stats", {})
        for stat_name, stat_value in life_stats.items():
            if isinstance(stat_value, (int, float)):
                metrics[f"life/{stat_name}"] = stat_value
        
        # Log location stats
        location_stats = info.get("location_stats", {})
        if "pos" in location_stats:
            pos = location_stats["pos"]
            if len(pos) >= 3:
                metrics.update({
                    "location/x": pos[0],
                    "location/y": pos[1],
                    "location/z": pos[2],
                    "location/distance_from_origin": np.sqrt(sum(p**2 for p in pos))
                })
        
        # Log planning metrics
        if planning_time is not None:
            metrics["planning/time_seconds"] = planning_time
        
        wandb.log(metrics)
    
    def log_episode_metrics(self, episode_metrics: EpisodeMetrics, task_id: str):
        """Log comprehensive episode metrics"""
        if not self.enabled:
            return
        
        episode_duration = episode_metrics.end_time - episode_metrics.start_time
        
        metrics = {
            "episode/duration_seconds": episode_duration,
            "episode/total_steps": episode_metrics.total_steps,
            "episode/total_reward": episode_metrics.total_reward,
            "episode/success": int(episode_metrics.success),
            "episode/inventory_diversity": episode_metrics.inventory_diversity,
            "episode/crafting_events": episode_metrics.crafting_events,
            "episode/mining_events": episode_metrics.mining_events,
            "episode/exploration_area": episode_metrics.exploration_area,
            "episode/planning_time": episode_metrics.planning_time,
            "episode/replanning_count": episode_metrics.replanning_count,
            "episode/goal_completion_rate": episode_metrics.goal_completion_rate,
            "episode/actions_per_second": episode_metrics.actions_per_second,
            "episode/reward_per_step": episode_metrics.reward_per_step,
            "episode/task_id": task_id,
        }
        
        if episode_metrics.time_to_first_success is not None:
            metrics["episode/time_to_first_success"] = episode_metrics.time_to_first_success
        
        wandb.log(metrics)
        
        # Log episode summary to wandb.run.summary for tracking across episodes
        wandb.run.summary.update({
            "latest_episode_success": episode_metrics.success,
            "latest_episode_reward": episode_metrics.total_reward,
            "latest_episode_steps": episode_metrics.total_steps,
            "total_episodes": wandb.run.summary.get("total_episodes", 0) + 1
        })
    
    def log_task_result(self, task_result: TaskResult, task_index: int):
        """Log individual task results"""
        if not self.enabled:
            return
        
        metrics = {
            "task/index": task_index,
            "task/success": int(task_result.success),
            "task/completion_time": task_result.completion_time,
            "task/total_steps": task_result.total_steps,
            "task/reward": task_result.reward,
            "task/completion_percentage": task_result.completion_percentage,
            "task/efficiency_score": task_result.efficiency_score,
            "task/planning_iterations": task_result.planning_iterations,
            "task/goal_changes": task_result.goal_changes,
            "task/type": task_result.task_type,
            "task/category": self._get_task_category(task_result.task_id),
        }
        
        # Log final inventory diversity
        metrics["task/final_inventory_diversity"] = len(task_result.final_inventory)
        
        # Log specific inventory items
        for item, count in task_result.final_inventory.items():
            metrics[f"task/final_inventory/{item}"] = count
        
        wandb.log(metrics)
        
        # Update task-level summaries
        wandb.run.summary.update({
            f"tasks/{task_result.task_type}_success_rate": self._calculate_category_success_rate(task_result.task_type),
            "tasks/total_completed": wandb.run.summary.get("tasks/total_completed", 0) + 1,
            "tasks/total_successful": wandb.run.summary.get("tasks/total_successful", 0) + (1 if task_result.success else 0)
        })
    
    def log_benchmark_summary(self, benchmark_suite: BenchmarkSuite):
        """Log comprehensive benchmark suite results"""
        if not self.enabled:
            return
        
        suite_duration = benchmark_suite.end_time - benchmark_suite.start_time
        
        # Main benchmark metrics
        metrics = {
            "benchmark/total_tasks": benchmark_suite.total_tasks,
            "benchmark/successful_tasks": benchmark_suite.successful_tasks,
            "benchmark/overall_success_rate": benchmark_suite.overall_success_rate,
            "benchmark/average_completion_time": benchmark_suite.average_completion_time,
            "benchmark/average_steps_per_task": benchmark_suite.average_steps_per_task,
            "benchmark/average_reward": benchmark_suite.average_reward,
            "benchmark/suite_duration": suite_duration,
        }
        
        # Category-specific metrics
        for category, results in [
            ("programmatic", benchmark_suite.programmatic_results),
            ("creative", benchmark_suite.creative_results),
            ("playthrough", benchmark_suite.playthrough_results)
        ]:
            if results:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                avg_steps = np.mean([r.total_steps for r in results])
                avg_reward = np.mean([r.reward for r in results])
                
                metrics.update({
                    f"benchmark/{category}_tasks": len(results),
                    f"benchmark/{category}_success_rate": success_rate,
                    f"benchmark/{category}_avg_steps": avg_steps,
                    f"benchmark/{category}_avg_reward": avg_reward,
                })
        
        # Tech tree progression
        for item, obtained in benchmark_suite.tech_tree_progression.items():
            metrics[f"tech_tree/{item}"] = int(obtained)
        
        # Survival metrics
        for metric, value in benchmark_suite.survival_metrics.items():
            metrics[f"survival/{metric}"] = value
        
        # Combat metrics
        for metric, value in benchmark_suite.combat_metrics.items():
            metrics[f"combat/{metric}"] = value
        
        wandb.log(metrics)
        
        # Update final summary
        wandb.run.summary.update({
            "final/overall_success_rate": benchmark_suite.overall_success_rate,
            "final/total_tasks": benchmark_suite.total_tasks,
            "final/successful_tasks": benchmark_suite.successful_tasks,
            "final/suite_duration": suite_duration,
        })
    
    def log_planning_metrics(self, 
                           plan: str,
                           goal_list: List[str],
                           selected_goal: Optional[str] = None,
                           planning_time: float = 0.0,
                           is_replanning: bool = False):
        """Log planning-specific metrics"""
        if not self.enabled:
            return
        
        metrics = {
            "planning/time_seconds": planning_time,
            "planning/goal_list_size": len(goal_list),
            "planning/is_replanning": int(is_replanning),
        }
        
        if selected_goal:
            metrics["planning/selected_goal"] = selected_goal
        
        wandb.log(metrics)
        
        # Log plan and goals as text
        wandb.log({
            "planning/plan_text": wandb.Html(f"<p><strong>Plan:</strong> {plan}</p>"),
            "planning/goals": wandb.Html(f"<ul>{''.join(f'<li>{goal}</li>' for goal in goal_list)}</ul>")
        })
    
    def log_media(self, 
                  step: int,
                  rgb_obs: Optional[np.ndarray] = None,
                  video_path: Optional[str] = None,
                  caption: Optional[str] = None):
        """Log images and videos"""
        if not self.enabled:
            return
        
        media_dict = {}
        
        if rgb_obs is not None:
            media_dict["obs/rgb_image"] = wandb.Image(
                rgb_obs, 
                caption=caption or f"Step {step}"
            )
        
        if video_path and Path(video_path).exists():
            media_dict["episode/video"] = wandb.Video(
                video_path,
                caption=caption or "Episode recording"
            )
        
        if media_dict:
            wandb.log(media_dict, step=step)
    
    def log_histogram(self, name: str, data: List[float], step: Optional[int] = None):
        """Log histogram data"""
        if not self.enabled or not data:
            return
        
        wandb.log({
            name: wandb.Histogram(data)
        }, step=step)
    
    def log_table(self, name: str, columns: List[str], data: List[List[Any]]):
        """Log table data"""
        if not self.enabled:
            return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})
    
    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "dataset"):
        """Log artifacts (datasets, models, etc.)"""
        if not self.enabled:
            return
        
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
    
    def _get_task_category(self, task_id: str) -> str:
        """Get task category for logging"""
        from benchmark_metrics import MinecraftTaskCategories
        return MinecraftTaskCategories.get_task_category(task_id)
    
    def _calculate_category_success_rate(self, task_type: str) -> float:
        """Calculate success rate for a specific task category"""
        # This would need access to all previous results
        # For now, return current session success rate from wandb.run.summary
        total_key = f"tasks/{task_type}_total"
        success_key = f"tasks/{task_type}_successful"
        
        total = wandb.run.summary.get(total_key, 0)
        successful = wandb.run.summary.get(success_key, 0)
        
        return (successful / max(total, 1)) * 100
    
    def finish(self):
        """Finish WandB logging"""
        if self.enabled:
            wandb.finish()
            print("[WandB] Logging finished")


class WandBIntegratedBenchmark:
    """
    Integration of BenchmarkMetrics with WandB logging
    """
    
    def __init__(self, 
                 benchmark_metrics: BenchmarkMetrics,
                 wandb_logger: WandBLogger):
        self.benchmark_metrics = benchmark_metrics
        self.wandb_logger = wandb_logger
        self.current_task_index = 0
    
    def start_episode(self, episode_id: str, task_id: str):
        """Start episode with both benchmark and WandB tracking"""
        self.benchmark_metrics.start_episode(episode_id, task_id)
        
        # Log episode start to WandB
        self.wandb_logger.log_step_metrics(
            step=0,
            reward=0.0,
            action={},
            inventory={},
            obs={},
            info={}
        )
    
    def log_step(self, 
                obs: Dict,
                action: Dict,
                reward: float,
                info: Dict,
                planning_time: Optional[float] = None):
        """Log step to both systems"""
        # Update benchmark metrics
        self.benchmark_metrics.update_step_metrics(obs, action, reward, info)
        
        # Log to WandB
        inventory = info.get('inventory', {})
        step = self.benchmark_metrics.current_episode_metrics.total_steps if self.benchmark_metrics.current_episode_metrics else 0
        
        self.wandb_logger.log_step_metrics(
            step=step,
            reward=reward,
            action=action,
            inventory=inventory,
            obs=obs,
            info=info,
            planning_time=planning_time
        )
    
    def log_planning(self, 
                    plan: str,
                    goal_list: List[str],
                    selected_goal: Optional[str] = None,
                    planning_time: float = 0.0,
                    is_replanning: bool = False):
        """Log planning information"""
        # Update benchmark metrics
        self.benchmark_metrics.update_planning_metrics(planning_time, is_replanning)
        
        # Log to WandB
        self.wandb_logger.log_planning_metrics(
            plan=plan,
            goal_list=goal_list,
            selected_goal=selected_goal,
            planning_time=planning_time,
            is_replanning=is_replanning
        )
    
    def end_episode(self, success: bool, final_inventory: Dict, task_id: str):
        """End episode with both systems"""
        # End benchmark episode
        episode_metrics = self.benchmark_metrics.end_episode(success, final_inventory)
        
        # Log to WandB
        self.wandb_logger.log_episode_metrics(episode_metrics, task_id)
        
        return episode_metrics
    
    def complete_task(self, 
                     task_id: str,
                     success: bool,
                     completion_time: float,
                     total_steps: int,
                     total_reward: float,
                     final_inventory: Dict,
                     planning_iterations: int = 0,
                     goal_changes: int = 0,
                     error_message: Optional[str] = None):
        """Complete task evaluation with both systems"""
        # Evaluate task with benchmark metrics
        task_result = self.benchmark_metrics.evaluate_task(
            task_id=task_id,
            success=success,
            completion_time=completion_time,
            total_steps=total_steps,
            total_reward=total_reward,
            final_inventory=final_inventory,
            planning_iterations=planning_iterations,
            goal_changes=goal_changes,
            error_message=error_message
        )
        
        # Log to WandB
        self.wandb_logger.log_task_result(task_result, self.current_task_index)
        self.current_task_index += 1
        
        return task_result
    
    def finish_benchmark(self):
        """Finish benchmark suite and log final results"""
        # Compute benchmark summary
        benchmark_suite = self.benchmark_metrics.compute_benchmark_summary()
        
        # Log to WandB
        self.wandb_logger.log_benchmark_summary(benchmark_suite)
        
        # Save benchmark results
        results_path = self.benchmark_metrics.save_results()
        
        # Log results file as artifact
        self.wandb_logger.log_artifact(
            str(results_path),
            "benchmark_results",
            "results"
        )
        
        # Print summary
        self.benchmark_metrics.print_summary()
        
        # Finish WandB
        self.wandb_logger.finish()
        
        return benchmark_suite
