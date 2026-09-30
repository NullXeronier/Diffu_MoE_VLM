"""
MC-Planner main entry point (Hydra).

    python main.py                                   # all tasks
    python main.py eval.single_task=true eval.task_name=obtain_stone_pickaxe
"""

import json
import warnings
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from diffu_moe_vlm.evaluator import Evaluator

warnings.filterwarnings('ignore')


@hydra.main(config_path="configs", config_name="defaults", version_base=None)
def main(cfg: OmegaConf) -> None:
    """Main entry point"""
    print("Starting MC-Planner evaluation...")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Initialize evaluator
    evaluator = Evaluator(cfg)
    
    try:
        # Run evaluation
        # Evaluation options live under `eval`; top-level keys are kept for backward compatibility
        eval_cfg = cfg.get('eval', {})
        single_task = eval_cfg.get('single_task') or cfg.get('single_task', False)
        if single_task:
            # Run single task
            task = cfg.get('task_name') or eval_cfg.get('task_name') or evaluator.task_list[0]
            result = evaluator.single_task_evaluate(task)
            
            # Save single task result
            output_file = Path(cfg.get('output_dir', '.')) / f'result_{task}.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Single task result saved to {output_file}")
            
        else:
            # Run all tasks
            results = evaluator.run_all_tasks()
            
            # Save results
            output_file = Path(cfg.get('output_dir', '.')) / 'results.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
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
            evaluator.wandb_logger.log({
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
