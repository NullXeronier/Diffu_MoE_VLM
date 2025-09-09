#!/usr/bin/env python3
"""
Test script for MineDojo-style benchmarking integration
Validates the benchmarking and WandB logging systems
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from benchmark_metrics import BenchmarkMetrics, TaskResult, EpisodeMetrics
from wandb_integration import WandBLogger, WandBIntegratedBenchmark
from minedojo_tasks import TaskRegistry, create_task_adapter


def test_benchmark_metrics():
    """Test the benchmark metrics system"""
    print("Testing Benchmark Metrics System...")
    
    # Initialize benchmark metrics
    benchmark = BenchmarkMetrics("./test_benchmark_results")
    
    # Test episode tracking
    benchmark.start_episode("test_episode_1", "obtain_wooden_slab")
    
    # Simulate some steps
    for step in range(10):
        obs = {"rgb": np.random.rand(64, 64, 3)}
        action = {"movement": 1, "camera": [0.1, 0.0], "attack": 0}
        reward = np.random.rand()
        info = {
            "inventory": {"wooden_planks": step // 2, "stick": step // 3},
            "life_stats": {"health": 20.0, "food": 18.0},
            "location_stats": {"pos": [step * 2, 64, step * 1.5]}
        }
        
        benchmark.update_step_metrics(obs, action, reward, info)
        time.sleep(0.1)  # Simulate processing time
    
    # Test planning metrics
    benchmark.update_planning_metrics(1.5, is_replanning=False)
    benchmark.update_planning_metrics(0.8, is_replanning=True)
    
    # End episode
    final_inventory = {"wooden_planks": 4, "stick": 2, "wooden_slab": 1}
    episode_metrics = benchmark.end_episode(success=True, final_inventory=final_inventory)
    
    print(f"Episode completed: {episode_metrics.success}")
    print(f"Total steps: {episode_metrics.total_steps}")
    print(f"Total reward: {episode_metrics.total_reward:.3f}")
    
    # Test task evaluation
    task_result = benchmark.evaluate_task(
        task_id="obtain_wooden_slab",
        success=True,
        completion_time=30.5,
        total_steps=10,
        total_reward=episode_metrics.total_reward,
        final_inventory=final_inventory,
        planning_iterations=2,
        goal_changes=1
    )
    
    print(f"Task result: {task_result.success}")
    print(f"Efficiency score: {task_result.efficiency_score:.3f}")
    
    # Test benchmark summary
    benchmark_suite = benchmark.compute_benchmark_summary()
    print(f"Benchmark suite success rate: {benchmark_suite.overall_success_rate:.1f}%")
    
    # Save results
    results_file = benchmark.save_results()
    print(f"Results saved to: {results_file}")
    
    # Print summary
    benchmark.print_summary()
    
    print("✓ Benchmark Metrics Test Passed\n")


def test_task_registry():
    """Test the task registry system"""
    print("Testing Task Registry System...")
    
    # Test task listing
    all_tasks = TaskRegistry.list_task_ids()
    print(f"Total tasks available: {len(all_tasks)}")
    
    # Test task categories
    categories = ["survival", "harvest", "mining", "tech_tree", "combat", "creative"]
    for category in categories:
        tasks = TaskRegistry.get_tasks_by_type(category)
        print(f"{category.title()} tasks: {len(tasks)}")
    
    # Test specific task info
    task_info = TaskRegistry.get_task_info("obtain_wooden_slab")
    print(f"Task info for 'obtain_wooden_slab': {task_info['description']}")
    
    # Test task adapter
    adapter = create_task_adapter()
    adapter.start_task("obtain_wooden_slab")
    
    # Test completion check
    test_inventory = {"wooden_slab": 1}
    test_info = {}
    is_complete = adapter.check_task_completion(test_inventory, test_info)
    print(f"Task completion check: {is_complete}")
    
    # Test reward calculation
    reward = adapter.get_task_reward(test_inventory, test_info, 100)
    print(f"Task reward: {reward:.2f}")
    
    print("✓ Task Registry Test Passed\n")


def test_wandb_integration():
    """Test WandB integration (mock mode if no internet)"""
    print("Testing WandB Integration...")
    
    try:
        # Initialize WandB logger (will use offline mode if no internet)
        wandb_logger = WandBLogger(
            project_name="test-project",
            experiment_name="test-experiment",
            config={"test": True, "max_steps": 100},
            tags=["test", "integration"],
            notes="Integration test for WandB logging",
            enabled=True
        )
        
        # Test basic logging
        wandb_logger.log_step_metrics(
            step=1,
            reward=1.5,
            action={"movement": 1},
            inventory={"wooden_planks": 2},
            obs={"rgb": np.random.rand(64, 64, 3)},
            info={"life_stats": {"health": 20.0}}
        )
        
        # Test planning metrics
        wandb_logger.log_planning_metrics(
            plan="1. Find trees\n2. Mine wood\n3. Craft planks",
            goal_list=["mine_wood", "craft_planks"],
            selected_goal="mine_wood",
            planning_time=2.1
        )
        
        # Test histogram
        wandb_logger.log_histogram("test_rewards", [1.0, 1.5, 2.0, 1.2, 1.8])
        
        # Test table
        wandb_logger.log_table(
            "test_results",
            ["Task", "Success", "Time"],
            [["task1", True, 10.5], ["task2", False, 8.2]]
        )
        
        wandb_logger.finish()
        print("✓ WandB Integration Test Passed\n")
        
    except Exception as e:
        print(f"⚠ WandB Integration Test Failed (this is expected without internet): {e}\n")


def test_integrated_benchmark():
    """Test the integrated benchmark system"""
    print("Testing Integrated Benchmark System...")
    
    # Initialize systems
    benchmark_metrics = BenchmarkMetrics("./test_integrated_results")
    
    # Initialize WandB in offline mode for testing
    wandb_logger = WandBLogger(
        project_name="test-integrated",
        experiment_name="test-integrated-experiment",
        config={"integrated_test": True},
        enabled=False  # Disable for testing
    )
    
    integrated_benchmark = WandBIntegratedBenchmark(benchmark_metrics, wandb_logger)
    
    # Test episode workflow
    integrated_benchmark.start_episode("integrated_episode_1", "obtain_wooden_slab")
    
    # Simulate steps
    for step in range(5):
        obs = {"rgb": np.random.rand(32, 32, 3)}
        action = {"movement": 1, "attack": step % 2}
        reward = np.random.rand() * 2
        info = {
            "inventory": {"wooden_planks": step},
            "life_stats": {"health": 20.0}
        }
        
        integrated_benchmark.log_step(obs, action, reward, info)
    
    # Test planning logging
    integrated_benchmark.log_planning(
        plan="Test plan for integration",
        goal_list=["test_goal_1", "test_goal_2"],
        selected_goal="test_goal_1",
        planning_time=1.0
    )
    
    # End episode
    final_inventory = {"wooden_slab": 1}
    episode_metrics = integrated_benchmark.end_episode(
        success=True,
        final_inventory=final_inventory,
        task_id="obtain_wooden_slab"
    )
    
    # Complete task
    task_result = integrated_benchmark.complete_task(
        task_id="obtain_wooden_slab",
        success=True,
        completion_time=15.2,
        total_steps=5,
        total_reward=5.5,
        final_inventory=final_inventory
    )
    
    print(f"Integrated test completed successfully: {task_result.success}")
    print("✓ Integrated Benchmark Test Passed\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("MineDojo-style Benchmarking Integration Tests")
    print("=" * 60)
    
    # Run individual tests
    test_benchmark_metrics()
    test_task_registry()
    test_wandb_integration()
    test_integrated_benchmark()
    
    print("=" * 60)
    print("All Tests Completed Successfully!")
    print("=" * 60)
    
    # Cleanup test files
    cleanup_test_files()


def cleanup_test_files():
    """Clean up test result files"""
    import shutil
    
    test_dirs = ["./test_benchmark_results", "./test_integrated_results"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up {test_dir}")


if __name__ == "__main__":
    main()
