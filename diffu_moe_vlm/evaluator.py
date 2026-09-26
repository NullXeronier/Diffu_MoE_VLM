"""
DEPS-style evaluation loop: Describe, Explain, Plan and Select.

For each task the planner proposes a plan, which is parsed into sub-goals.
The selector picks the next sub-goal, the controller acts on it until it is
reached, and failures (invalid actions or a goal that takes too long) trigger
a replan with a description of the current inventory and the failure.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from .benchmark_metrics import BenchmarkMetrics
from .controller import GoalController
from .core import data_manager, goal_item, task_registry
from .env import MineDojoEnv
from .planner import Planner
from .selector import HorizonSelector, Selector
from .wandb_integration import WandBIntegratedBenchmark, WandBLogger


class Evaluator:
    """Main evaluator class for MC-Planner experiments"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        eval_cfg = cfg.get('eval', {})
        wandb_cfg = cfg.get('wandb', {})

        self.wandb_logger = WandBLogger(
            project_name=wandb_cfg.get('project', 'diffu-moe-vlm-minecraft'),
            experiment_name=wandb_cfg.get('experiment_name', None),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=wandb_cfg.get('tags', ['minecraft', 'vlm', 'planning']),
            notes=wandb_cfg.get('notes', None),
            enabled=wandb_cfg.get('enabled', True)
        )
        self.benchmark_metrics = BenchmarkMetrics(
            output_dir=str(Path(cfg.get('output_dir', '.')) / 'benchmark_results')
        )
        self.integrated_benchmark = WandBIntegratedBenchmark(self.benchmark_metrics, self.wandb_logger)

        # Environment
        env_cfg = cfg.get('env', {})
        self.max_steps = eval_cfg.get('max_steps', 1000)
        self.env = MineDojoEnv(
            name=eval_cfg.get('env_name', 'Plains'),
            img_size=tuple(cfg.simulator.resolution),
            rgb_only=cfg.simulator.get('rgb_only', False),
            max_steps=self.max_steps,
            action_failure_prob=env_cfg.get('action_failure_prob', 0.0),
        )
        self.seed = env_cfg.get('seed', None)
        self._seeded = False

        # Tasks
        data_cfg = cfg.get('data', {})
        enabled_tasks = data_cfg.get('tasks', {}).get('enabled') if data_cfg else None
        self.task_list = list(enabled_tasks) if enabled_tasks else task_registry.list_tasks()
        self.task_info = data_manager.load_task_info()

        # Planner / selector / controller
        llm_cfg = cfg.get('llm', {})
        self.planner = Planner(
            api_base=llm_cfg.get('api_base'),
            model=llm_cfg.get('model'),
            api_key=llm_cfg.get('api_key'),
            enabled=llm_cfg.get('enabled', True),
            timeout=llm_cfg.get('timeout', 30),
        )
        self.selector = self._build_selector(cfg.get('goal_model', {}))
        controller_cfg = cfg.get('controller', {})
        self.controller = GoalController(auto_prerequisites=controller_cfg.get('auto_prerequisites', True))

        # Replanning budget
        self.max_replans = eval_cfg.get('max_replans', 5)
        self.replan_after_failures = eval_cfg.get('replan_after_failures', 3)
        self.max_goal_steps = eval_cfg.get('max_goal_steps', 50)
        self.max_candidates = cfg.get('goal_model', {}).get('max_candidates', 5)

        self.log_images = wandb_cfg.get('log_images', False)
        self.log_frequency = wandb_cfg.get('log_frequency', 10)

        print(f"[Progress] Tasks: {self.task_list}")
        print(f"[Progress] Selector: {type(self.selector).__name__}({self.selector.strategy}), "
              f"auto_prerequisites={self.controller.auto_prerequisites}")
        print(f"[Progress] WandB logging enabled: {self.wandb_logger.enabled}")

    @staticmethod
    def _build_selector(goal_cfg) -> Selector:
        strategy = goal_cfg.get('strategy', 'plan_order')
        priorities = goal_cfg.get('priorities', None)
        priorities = dict(priorities) if priorities else None
        if goal_cfg.get('use_horizon_planning', False):
            return HorizonSelector(horizon=goal_cfg.get('horizon', 3), strategy=strategy, priorities=priorities)
        return Selector(strategy=strategy, priorities=priorities)

    # ------------------------------------------------------------------

    def reset(self, task: str):
        """Reset environment and agent modules for a new task"""
        seed = None
        if not self._seeded:
            seed, self._seeded = self.seed, True
        obs, info = self.env.reset(seed=seed, options={'task': task})
        self.planner.reset()
        self.selector.reset()
        print(f"[Reset] Starting task: {task}")
        return obs, info

    def _replan(self, question: str, inventory: Dict[str, int], failure_desc: str) -> Tuple[str, List[str]]:
        start = time.time()
        plan = self.planner.replan(
            question,
            self.planner.generate_inventory_description(inventory),
            inventory=inventory,
            failure_desc=failure_desc,
        )
        goal_list = self.planner.generate_goal_list(plan)
        self.integrated_benchmark.log_planning(
            plan=plan, goal_list=goal_list, planning_time=time.time() - start, is_replanning=True
        )
        print(f"[Replan] {failure_desc} -> {goal_list}")
        return plan, goal_list

    def single_task_evaluate(self, task: Optional[str] = None) -> Dict[str, Any]:
        """Run one episode of the DEPS loop on a task"""
        task = task or self.task_list[0]
        task_def = task_registry.get_task(task)
        if task_def is None:
            raise ValueError(f"Unknown task: {task}")
        target = task_def['target']

        obs, info = self.reset(task)
        inventory = info['inventory']
        group = self.task_info.get(task, {}).get('group', task_def['group'])
        question = f"How to obtain {target.replace('_', ' ')}?"

        episode_id = f"{task}_{int(time.time())}"
        self.integrated_benchmark.start_episode(episode_id, task)

        # Plan
        start = time.time()
        plan = self.planner.initial_planning(group=group, task_question=question, target=target)
        goal_list = self.planner.generate_goal_list(plan)
        self.integrated_benchmark.log_planning(
            plan=plan, goal_list=goal_list, planning_time=time.time() - start, is_replanning=False
        )
        print(f"[Planning] Goal list: {goal_list}")
        initial_goals = list(goal_list)

        success = bool(info.get('task_success'))
        total_reward = 0.0
        steps = 0
        replans = 0
        goal_changes = 0
        # Goal instances complete when their item is obtained (event-based), so a plan
        # can require e.g. "mine cobblestone" six times and old stock never counts
        goal_done = [False] * len(goal_list)
        current_goal: Optional[str] = None
        goal_steps = 0
        consecutive_failures = 0
        episode_start = time.time()

        while not success and steps < self.max_steps:
            pending = list(dict.fromkeys(g for g, done in zip(goal_list, goal_done) if not done))
            if current_goal not in pending:
                current_goal = None

            # Select
            if current_goal is None:
                if not pending:
                    if replans >= self.max_replans:
                        break
                    plan, goal_list = self._replan(question, inventory, "plan finished but the target is missing")
                    goal_done = [False] * len(goal_list)
                    replans += 1
                    continue
                candidates = self.selector.generate_candidate_goal_list(pending, inventory, self.max_candidates)
                current_goal = self.selector.horizon_select(candidates, inventory)
                goal_steps = 0
                goal_changes += 1
                self.wandb_logger.log_planning_metrics(
                    plan=plan, goal_list=goal_list, selected_goal=current_goal,
                    planning_time=0.0, is_replanning=False
                )

            # Act
            action = self.controller.next_action(current_goal, inventory)
            if action is None:
                print(f"[Controller] Unknown goal {current_goal!r}, skipping")
                goal_done = [done or g == current_goal for g, done in zip(goal_list, goal_done)]
                current_goal = None
                continue

            previous_inventory = inventory
            obs, reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            goal_steps += 1
            total_reward += reward
            inventory = info['inventory']
            self.integrated_benchmark.log_step(obs=obs, action=action, reward=reward, info=info, planning_time=0.0)
            if self.log_images and steps % self.log_frequency == 0 and 'rgb' in obs:
                self.wandb_logger.log_media(step=steps, rgb_obs=obs['rgb'],
                                            caption=f"Step {steps} - Goal: {current_goal}")

            # Mark the earliest pending instance of every item obtained this step
            for item, count in inventory.items():
                if count > previous_inventory.get(item, 0):
                    for i, g in enumerate(goal_list):
                        if not goal_done[i] and goal_item(g) == item:
                            goal_done[i] = True
                            if g == current_goal:
                                current_goal = None
                            break

            if info['task_success']:
                success = True
                break
            if truncated:
                break

            # Explain & replan on repeated failures or a stuck goal
            consecutive_failures = 0 if info['action_success'] else consecutive_failures + 1
            stuck = current_goal is not None and goal_steps >= self.max_goal_steps
            if consecutive_failures >= self.replan_after_failures or stuck:
                if replans >= self.max_replans:
                    break
                failure_desc = (info['action_error'] if consecutive_failures
                                else f"goal {current_goal} not reached after {goal_steps} steps")
                plan, goal_list = self._replan(question, inventory, failure_desc)
                goal_done = [False] * len(goal_list)
                replans += 1
                current_goal = None
                consecutive_failures = 0

        completion_time = time.time() - episode_start
        print(f"[Result] {task}: success={success} steps={steps} replans={replans}")

        episode_metrics = self.integrated_benchmark.end_episode(
            success=success, final_inventory=inventory, task_id=task
        )
        task_result = self.integrated_benchmark.complete_task(
            task_id=task,
            success=success,
            completion_time=completion_time,
            total_steps=steps,
            total_reward=total_reward,
            final_inventory=inventory,
            planning_iterations=episode_metrics.replanning_count,
            goal_changes=goal_changes,
        )

        return {
            'task': task,
            'target': target,
            'success': success,
            'steps': steps,
            'replans': replans,
            'total_reward': total_reward,
            'completion_time': completion_time,
            'initial_goals': initial_goals,
            'final_inventory': inventory,
            'efficiency_score': task_result.efficiency_score,
        }

    def run_all_tasks(self) -> List[Dict[str, Any]]:
        """Run evaluation on all configured tasks"""
        results = []
        task_failures = 0

        print(f"[Benchmark] Starting evaluation of {len(self.task_list)} tasks")
        for i, task in enumerate(self.task_list):
            try:
                print(f"[Benchmark] Evaluating task {i + 1}/{len(self.task_list)}: {task}")
                results.append(self.single_task_evaluate(task))
                successful_tasks = sum(1 for r in results if r.get('success', False))
                self.wandb_logger.log({
                    "benchmark/current_task_index": i + 1,
                    "benchmark/current_success_rate": (successful_tasks / len(results)) * 100,
                    "benchmark/tasks_completed": len(results),
                    "benchmark/tasks_remaining": len(self.task_list) - len(results)
                })
            except Exception as e:
                print(f"[Error] Failed to evaluate task {task}: {e}")
                results.append({
                    'task': task, 'success': False, 'error': str(e), 'steps': 0,
                    'total_reward': 0.0, 'completion_time': 0.0, 'final_inventory': {}
                })
                task_failures += 1
                self.wandb_logger.log({"errors/task_failures": task_failures, "errors/latest_error": str(e)})

        self.integrated_benchmark.finish_benchmark()

        if results:
            successful_tasks = sum(1 for r in results if r.get('success', False))
            total_steps = sum(r.get('steps', 0) for r in results)
            print("\n[Summary] Benchmark completed!")
            print(f"[Summary] Tasks completed: {successful_tasks}/{len(results)}")
            print(f"[Summary] Overall success rate: {successful_tasks / len(results) * 100:.2f}%")
            print(f"[Summary] Average steps per task: {total_steps / len(results):.2f}")
            for r in results:
                print(f"[Summary]   {r['task']:<22} success={r.get('success')} "
                      f"steps={r.get('steps')} replans={r.get('replans', 0)}")
        return results
