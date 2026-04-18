# MC-Planner DEPS Simulator

This project Simulates Open Ended Env
## Updated: Exceed consume metric(EAT_PLANT) in PPO Actor with IMU data 
https://wandb.ai/catted/Craftax_Baselines?nw=nwusercatted
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
├── src/
│   ├── minedojo_core/          # Core minedojo components (data, sim, tasks)
│   ├── gymnasium_env/          # Gymnasium environment wrapper
│   ├── models/                 # Model implementations
│   ├── utils/                  # Utility functions
│   └── __init__.py
├── configs/                    # Configuration files
├── data/                       # Data files and prompts
├── main.py                     # Main execution script
├── planner.py                  # Planning module
├── selector.py                 # Selection module
├── controller.py               # Controller module
├── requirements.txt            # Python dependencies
└── README.md
```

## Features

- **Java Dependency Removal**: Minimized Java dependencies outside Minecraft MDK
- **Gymnasium Integration**: Migrated from MineDojo to gymnasium environment
- **Modular Design**: Separated core components for better maintainability
- **Configuration-based**: Experiment configuration through YAML files
- **Optimized Execution**: Streamlined for research and experimentation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

```bash
# Run with default configuration
python main.py

# Run single task
python main.py eval.single_task=true eval.task_name=obtain_wooden_slab

# Run with custom configuration
python main.py --config-path configs --config-name custom

# Test core functionality
python -c "
import sys; sys.path.append('src')
from src.gymnasium_env import MineDojoEnv
from planner import Planner
from selector import Selector
print('All modules working correctly')
"
```

## Quick Start

1. **Clone and install**:

   ```bash
   git clone <repository>
   cd Simulator-master
   pip install -r requirements.txt
   ```

2. **Test the installation**:

   ```bash
   python -c "from src.gymnasium_env import MineDojoEnv; print('✓ Installation successful')"
   ```

3. **Run a simple task**:

   ```bash
   python main.py eval.single_task=true eval.task_name=obtain_wooden_slab
   ```

## Local LLM Setup

For planning functionality, you can use a local LLM:

```bash
# Set environment variables
export LLM_API_BASE="http://localhost:8000/v1"
export LLM_MODEL="local-llama3"
export LLM_API_KEY="DUMMY"

# Or edit data/openai_keys.txt
```

## Components

### Core Modules

- **main.py**: Main entry point and experiment orchestration
- **planner.py**: LLM-based planning with local model support
- **selector.py**: Goal selection and horizon planning
- **controller.py**: Action execution and environment interaction

### Configuration

All experiments can be configured through YAML files in the `configs/` directory.

### Data

The `data/` directory contains:

- Task definitions and prompts
- Goal mappings and libraries
- Pre-computed embeddings

## Development

This project is designed for research in multi-task agents using large language models in Minecraft environments.
