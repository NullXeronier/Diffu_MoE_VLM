# MineDojo-style Benchmarking Integration for Diffu_MoE_VLM

This document describes the comprehensive benchmarking and logging system integrated into the Diffu_MoE_VLM project, based on MineDojo's evaluation framework.

## Overview

The integration provides:

1. **Comprehensive Metrics Collection** - Following MineDojo's benchmarking suite structure
2. **WandB Integration** - Automated logging to Weights & Biases for experiment tracking
3. **Task Registry** - MineDojo-compatible task definitions and management
4. **Performance Analytics** - Detailed analysis of agent performance across different task categories

## Components

### 1. Benchmark Metrics (`benchmark_metrics.py`)

Implements MineDojo-style metrics collection:

- **TaskResult**: Individual task evaluation results
- **EpisodeMetrics**: Per-episode performance tracking
- **BenchmarkSuite**: Comprehensive benchmark suite results
- **MinecraftTaskCategories**: Task categorization system

Key metrics tracked:
- Success rates by task category (survival, harvest, mining, tech tree, combat, creative)
- Planning efficiency (iterations, replanning frequency)
- Resource utilization (inventory diversity, crafting/mining events)
- Exploration metrics (area covered)
- Performance metrics (steps per task, reward per step)

### 2. WandB Integration (`wandb_integration.py`)

Provides comprehensive experiment logging:

- **WandBLogger**: Core logging functionality
- **WandBIntegratedBenchmark**: Integrated benchmark + logging system

Features:
- Real-time step logging
- Episode-level metrics aggregation
- Task completion tracking
- Media logging (screenshots, videos)
- Custom metric definitions
- Artifact management

### 3. Task Registry (`minedojo_tasks.py`)

MineDojo-compatible task system:

- **TaskRegistry**: Centralized task definitions
- **MinedojoTaskAdapter**: Task execution adapter
- **Programmatic Tasks**: 15+ tasks across survival, harvest, mining, tech tree, and combat
- **Creative Tasks**: 5+ building and creativity challenges
- **Playthrough Task**: Ultimate challenge (defeat Ender Dragon)

## Usage

### Basic Integration

```python
from benchmark_metrics import BenchmarkMetrics
from wandb_integration import WandBLogger, WandBIntegratedBenchmark
from minedojo_tasks import create_task_adapter

# Initialize benchmarking
benchmark_metrics = BenchmarkMetrics("./benchmark_results")
wandb_logger = WandBLogger("my-minecraft-project", config=my_config)
benchmark = WandBIntegratedBenchmark(benchmark_metrics, wandb_logger)

# Start episode
benchmark.start_episode("episode_1", "obtain_wooden_slab")

# Log steps
for step in range(max_steps):
    # ... environment step ...
    benchmark.log_step(obs, action, reward, info)

# End episode
benchmark.end_episode(success=True, final_inventory=inventory, task_id="obtain_wooden_slab")

# Complete task
benchmark.complete_task(
    task_id="obtain_wooden_slab",
    success=True,
    completion_time=120.5,
    total_steps=500,
    total_reward=85.2,
    final_inventory=inventory
)

# Finish benchmark suite
benchmark.finish_benchmark()
```

### Configuration

Add to your `configs/defaults.yaml`:

```yaml
# Benchmarking settings
benchmark_output_dir: "./benchmark_results"
benchmark_enabled: true

# WandB logging settings
wandb_enabled: true
wandb_project: "diffu-moe-vlm-minecraft"
wandb_tags: ["minecraft", "planning", "vlm", "benchmark"]
wandb_notes: "MineDojo-style benchmarking for Diffu_MoE_VLM project"
```

### Task Definitions

Tasks are organized by category:

**Survival Tasks:**
- `survive_1_day`: Survive for 1 Minecraft day
- `survive_3_days`: Survive for 3 Minecraft days

**Harvest Tasks:**
- `obtain_wooden_slab`: Craft wooden slab from planks
- `obtain_stone_stairs`: Craft stone stairs from cobblestone
- `obtain_painting`: Craft painting from sticks and wool
- `harvest_milk`: Obtain milk from cow
- `harvest_wool`: Shear wool from sheep

**Mining Tasks:**
- `mine_cobblestone`: Mine cobblestone with pickaxe
- `mine_iron_ore`: Mine iron ore with stone pickaxe
- `mine_diamond`: Mine diamond with iron pickaxe

**Tech Tree Tasks:**
- `craft_wooden_pickaxe`: Craft wooden pickaxe
- `craft_stone_pickaxe`: Craft stone pickaxe
- `craft_iron_pickaxe`: Craft iron pickaxe
- `craft_diamond_pickaxe`: Craft diamond pickaxe

**Combat Tasks:**
- `combat_zombie`: Defeat zombie with sword
- `combat_skeleton`: Defeat skeleton with sword and shield
- `hunt_pig`: Hunt pig for food

**Creative Tasks:**
- `build_house`: Build simple house
- `build_castle`: Build castle with towers
- `build_bridge`: Build bridge across water
- `create_art`: Create pixel art
- `build_farm`: Build functioning farm

**Playthrough Task:**
- `defeat_ender_dragon`: Ultimate challenge

## Metrics Collected

### Episode-Level Metrics

- **Duration**: Episode time in seconds
- **Steps**: Total environment steps
- **Reward**: Cumulative reward
- **Success**: Task completion status
- **Inventory Diversity**: Number of unique items obtained
- **Crafting/Mining Events**: Action counts
- **Exploration Area**: Area covered in blocks²
- **Planning Metrics**: Planning time, replanning count
- **Efficiency**: Actions per second, reward per step

### Task-Level Metrics

- **Success Rate**: By task and category
- **Completion Time**: Average time to complete
- **Steps to Completion**: Efficiency metric
- **Planning Iterations**: Number of planning cycles
- **Goal Changes**: Planning adaptation frequency

### Benchmark Suite Metrics

- **Overall Performance**: Across all task categories
- **Category Breakdown**: Performance by task type
- **Tech Tree Progression**: Which tools/items were crafted
- **Survival Stats**: Health, hunger, survival time
- **Combat Stats**: Monsters defeated, damage metrics

## WandB Dashboard Features

### Real-time Monitoring

- Step-by-step progress tracking
- Live performance metrics
- Resource utilization graphs
- Planning efficiency charts

### Episode Analysis

- Episode success/failure tracking
- Completion time distributions
- Reward progression analysis
- Inventory diversity trends

### Task Performance

- Success rates by category
- Difficulty-based performance analysis
- Planning strategy effectiveness
- Comparative analysis across tasks

### Media Logging

- Screenshots at key moments
- Episode recordings (if enabled)
- Planning visualizations
- Performance summary tables

## Output Files

### Benchmark Results

```
benchmark_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # Complete results
├── task_registry.json                       # Task definitions
├── task_mapping.json                        # Task descriptions
└── tasks_*.json                            # Category-specific tasks
```

### WandB Artifacts

- Benchmark result files
- Task completion videos
- Performance summary reports
- Model checkpoints (if applicable)

## Advanced Features

### Custom Metrics

Define custom metrics in WandB:

```python
wandb.define_metric("custom/metric", step_metric="episode/step")
wandb.log({"custom/metric": value})
```

### Histogram Logging

Track distribution of values:

```python
wandb_logger.log_histogram("rewards", reward_history)
```

### Table Logging

Log structured data:

```python
wandb_logger.log_table("results", ["Task", "Success", "Time"], data)
```

### Artifact Management

Save important files:

```python
wandb_logger.log_artifact("model.pt", "model_checkpoint", "model")
```

## Integration with Main Evaluation

The main evaluator (`main.py`) automatically integrates the benchmarking system:

1. Initializes benchmark and WandB systems
2. Tracks all episodes and tasks
3. Logs comprehensive metrics
4. Generates final reports
5. Uploads results as artifacts

## Performance Considerations

- **Logging Frequency**: Adjustable step logging to avoid overwhelming WandB
- **Media Logging**: Configurable frequency for screenshots/videos
- **Metric Aggregation**: Efficient batch logging for better performance
- **Storage Management**: Automatic cleanup of old benchmark files

## Dependencies

Required packages (already in `requirements.txt`):
- `wandb>=0.15.0`
- `tensorboard>=2.12.0`
- `numpy>=1.21.0`
- `torch>=2.0.0`

## Configuration Examples

### Minimal Configuration

```yaml
wandb_enabled: true
wandb_project: "my-project"
benchmark_enabled: true
```

### Full Configuration

```yaml
# Benchmarking
benchmark_output_dir: "./results/benchmarks"
benchmark_enabled: true

# WandB
wandb_enabled: true
wandb_project: "diffu-moe-vlm-minecraft"
wandb_tags: ["experiment", "v1.0", "minecraft"]
wandb_notes: "Baseline evaluation with improved planning"
experiment_name: "baseline_v1"

# Task selection
eval:
  single_task: false  # Run all tasks
  task_name: "obtain_wooden_slab"  # If single_task: true
  max_steps: 1000
```

## Troubleshooting

### WandB Connection Issues

```python
# Disable WandB if having connection issues
wandb_enabled: false
```

### Memory Issues

```python
# Reduce logging frequency
log_frequency: 100  # Log every 100 steps instead of every step
```

### Storage Issues

```python
# Limit benchmark history
max_benchmark_files: 10  # Keep only last 10 benchmark files
```

This benchmarking integration provides comprehensive evaluation capabilities matching MineDojo's standards while being specifically tailored for the Diffu_MoE_VLM project's needs.
