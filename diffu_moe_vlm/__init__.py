"""
Diffu_MoE_VLM: DEPS-style planning agent on a symbolic Minecraft tech tree.

Submodules are imported lazily so that light-weight components (env, core,
controller) do not pull in optional heavy dependencies such as torch or wandb.
"""

__version__ = "0.2.0"
