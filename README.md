# MC-Planner DEPS Simulator

This project Simulates Open Ended Env
## Updated: Exceed consume metric(EAT_PLANT, MAKE...) in PPO Actor with IMU data 
this - with 3d trajectory time embedding
<img width="906" height="397" alt="그림2" src="https://github.com/user-attachments/assets/2902ae83-2304-4ed9-914f-4ebd78411d2e" />

trajectory
<img width="743" height="786" alt="image" src="https://github.com/user-attachments/assets/b5e58bed-da63-49ea-bbd7-7b585fd69fac" />


original paper: AN EFFICIENT OPEN WORLD ENVIRONMENT FOR MULTI-AGENT
SOCIAL LEARNING

<img width="1565" height="534" alt="image" src="https://github.com/user-attachments/assets/5954c521-06ba-44ef-b0ed-739e4570f083" />


## Project Structure

```text
.
├── diffu_moe_vlm/
│   ├── core.py               # Tech tree (recipes, tools), tasks, plan parsing, data loading
│   ├── env.py                # Symbolic Minecraft environment (Gymnasium API)
│   ├── planner.py            # LLM planner with rule-based fallback (DEPS: describe/explain/plan)
│   ├── selector.py           # Sub-goal selection (plan_order, priority, dependency, horizon)
│   ├── controller.py         # Goal-conditioned controller producing macro actions
│   ├── evaluator.py          # DEPS evaluation loop and benchmark bookkeeping
│   ├── benchmark_metrics.py  # MineDojo-style metrics
│   ├── wandb_integration.py  # Weights & Biases logging
│   ├── fp8_utils.py          # Optional FP8 / TensorRT-LLM support
│   └── data/                 # Goal library, task info, prompts
├── configs/                  # Hydra configuration
├── tests/                    # pytest suite
├── main.py                   # Entry point
└── pyproject.toml
```

## How the baseline works

1. **Environment** (`env.py`): a symbolic tech-tree world. Actions are macros such as
   `{'type': 'mine', 'item': 'wood'}` or `{'type': 'craft', 'item': 'stick'}`; they succeed only when
   the required tools and ingredients are in the inventory (e.g. cobblestone needs a wooden pickaxe,
   an iron ingot needs a furnace and coal). There is no 3D world yet: RGB/depth observations are blank
   frames kept for interface compatibility.
2. **Planner** (`planner.py`): asks an OpenAI-compatible LLM for a plan. If the LLM is disabled or
   unreachable, a rule-based planner derives the plan from the tech tree (quantity aware, e.g.
   `Mine cobblestone x6`). Plans are parsed into sub-goals such as `mine_wood`, `obtain_stick`.
3. **Selector** (`selector.py`): picks the next sub-goal among pending ones.
4. **Controller** (`controller.py`): turns the sub-goal into actions. With
   `controller.auto_prerequisites=true` it gathers missing tools/ingredients itself; with `false` it
   only attempts the sub-goal, so success depends on plan quality.
5. **Replanning** (`evaluator.py`): repeated action failures or a stuck sub-goal trigger a replan
   with the current inventory and the failure message.

Baseline (rule-based planner, default config): 6/6 default tasks succeed, e.g. `obtain_wooden_slab`
in 3 steps and `mine_diamond` in 34 steps.

## Installation

```bash
pip install -e .            # core (CPU only, no torch needed)
pip install -e ".[dev]"     # + pytest
pip install -e ".[ml]"      # + torch / transformers for model and FP8 code
```

## Usage

```bash
# All default tasks
python main.py

# Single task
python main.py eval.single_task=true eval.task_name=obtain_stone_pickaxe

# Offline, without an LLM server or WandB
python main.py llm.enabled=false wandb.enabled=false

# Ablations
python main.py controller.auto_prerequisites=false   # plan quality must carry the task
python main.py env.action_failure_prob=0.3           # noisy low-level control, exercises replanning
python main.py goal_model.strategy=priority          # plan_order | priority | random | round_robin | dependency

# Tests
pytest
```

Results are written to `output_dir` (default `./outputs`): `results.json` or `result_<task>.json`.

## Local LLM Setup

The planner talks to an OpenAI-compatible `/chat/completions` endpoint. Configure it in
`configs/defaults.yaml` (`llm:` section) or with environment variables, which take precedence:

```bash
export LLM_API_BASE="http://localhost:8000/v1"
export LLM_MODEL="local-llama3"
export LLM_API_KEY="DUMMY"
# Or copy .env.example to .env (never commit API keys)
```

## Data

`diffu_moe_vlm/data/` contains the goal library, goal mappings, task info and the prompt templates
used by the planner.

## Development

This project is designed for research in multi-task agents using large language models in Minecraft
environments. Next steps: a real (pixel-based) Minecraft backend, and VLM / MoE / diffusion policy
modules in place of the scripted controller.
