#!/usr/bin/env python3
"""
Test script for WandB integration in main.py
"""

import os
import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_wandb_integration():
    """Test WandB integration"""
    print("Testing WandB integration...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set environment variables for testing
        os.environ["WANDB_MODE"] = "offline"  # Run in offline mode for testing
        
        # Test configuration
        test_config = {
            'wandb': {
                'enabled': True,
                'project': 'test-diffu-moe-vlm',
                'experiment_name': 'test_run',
                'tags': ['test', 'minecraft'],
                'notes': 'Test run for WandB integration',
                'log_images': False,  # Disable image logging for test
                'mode': 'offline'
            },
            'eval': {
                'env_name': 'Plains',
                'max_steps': 10  # Very short test
            },
            'simulator': {
                'resolution': [64, 64]  # Small resolution for test
            },
            'goal_model': {
                'freq': 5,
                'queue_size': 3
            },
            'record': {
                'frames': False
            },
            'single_task': True,
            'task_name': 'obtain_wooden_slab',
            'output_dir': temp_dir
        }
        
        try:
            # Import and test
            from omegaconf import OmegaConf
            cfg = OmegaConf.create(test_config)
            
            # Test imports
            from wandb_integration import WandBLogger, WandBIntegratedBenchmark
            from benchmark_metrics import BenchmarkMetrics
            
            print("✓ Successfully imported WandB integration modules")
            
            # Test WandB logger initialization
            logger = WandBLogger(
                project_name=test_config['wandb']['project'],
                experiment_name=test_config['wandb']['experiment_name'],
                config=test_config,
                tags=test_config['wandb']['tags'],
                notes=test_config['wandb']['notes'],
                enabled=True
            )
            
            print("✓ Successfully initialized WandB logger")
            
            # Test basic logging
            logger.log_step_metrics(
                step=0,
                reward=0.0,
                action={'movement': 0},
                inventory={},
                obs={'rgb': None},
                info={}
            )
            
            print("✓ Successfully logged test metrics")
            
            # Clean up
            logger.finish()
            print("✓ Successfully finished WandB session")
            
            print("\n🎉 WandB integration test passed!")
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all required modules are available")
            return False
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

if __name__ == "__main__":
    success = test_wandb_integration()
    sys.exit(0 if success else 1)
