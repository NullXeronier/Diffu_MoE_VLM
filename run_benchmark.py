#!/usr/bin/env python3
"""
Quick Start Script for MineDojo-style Benchmarking
Usage: python run_benchmark.py [--config CONFIG] [--task TASK] [--no-wandb]
"""

import argparse
import sys
from pathlib import Path
import hydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import main


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run MineDojo-style benchmarking for Diffu_MoE_VLM"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="benchmark",
        help="Configuration file name (default: benchmark)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Single task to run (if not specified, runs all tasks)"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Tags for WandB logging"
    )
    
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes for this experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test configuration without running evaluation"
    )
    
    return parser.parse_args()


def run_benchmark_with_args(args):
    """Run benchmark with command line arguments"""
    
    # Build Hydra overrides
    overrides = []
    
    if args.task:
        overrides.extend([
            "eval.single_task=true",
            f"eval.task_name={args.task}"
        ])
    
    if args.no_wandb:
        overrides.append("wandb.enabled=false")
    
    if args.experiment_name:
        overrides.append(f"experiment_name={args.experiment_name}")
    
    if args.output_dir:
        overrides.append(f"output.base_dir={args.output_dir}")
    
    if args.tags:
        tag_str = "[" + ",".join(f'"{tag}"' for tag in args.tags) + "]"
        overrides.append(f"wandb.tags={tag_str}")
    
    if args.notes:
        overrides.append(f'wandb.notes="{args.notes}"')
    
    # Set config name
    config_name = args.config
    
    if args.dry_run:
        print("DRY RUN MODE - Configuration Preview:")
        print("=" * 50)
        print(f"Config: {config_name}")
        print(f"Overrides: {overrides}")
        
        # Load and display config
        from hydra import initialize, compose
        
        with initialize(config_path="configs", version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides)
            print("Final Configuration:")
            print(OmegaConf.to_yaml(cfg))
        
        return
    
    # Create temporary Hydra config
    @hydra.main(config_path="configs", config_name=config_name, version_base=None)
    def hydra_main(cfg):
        return main(cfg)
    
    # Override sys.argv for Hydra
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + overrides
    
    try:
        result = hydra_main()
        return result
    finally:
        sys.argv = original_argv


def quick_examples():
    """Print quick usage examples"""
    print("Quick Start Examples:")
    print("=" * 50)
    print()
    print("1. Run full benchmark suite:")
    print("   python run_benchmark.py")
    print()
    print("2. Run single task:")
    print("   python run_benchmark.py --task obtain_wooden_slab")
    print()
    print("3. Run without WandB logging:")
    print("   python run_benchmark.py --no-wandb")
    print()
    print("4. Custom experiment:")
    print("   python run_benchmark.py --experiment-name my_experiment --tags test baseline")
    print()
    print("5. Test configuration:")
    print("   python run_benchmark.py --dry-run")
    print()
    print("6. Creative tasks only:")
    print("   python run_benchmark.py --config creative_benchmark")
    print()


def main_cli():
    """Main CLI entry point"""
    args = parse_args()
    
    if len(sys.argv) == 1:
        print("MineDojo-style Benchmarking for Diffu_MoE_VLM")
        print("=" * 50)
        quick_examples()
        return
    
    print("Starting MineDojo-style Benchmarking...")
    print(f"Configuration: {args.config}")
    print(f"Single task: {args.task or 'All tasks'}")
    print(f"WandB enabled: {not args.no_wandb}")
    print()
    
    try:
        result = run_benchmark_with_args(args)
        
        if not args.dry_run:
            print()
            print("Benchmarking completed successfully!")
            if result:
                print(f"Results: {result}")
    
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
