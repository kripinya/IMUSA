"""
Configuration management for IMUSA.

Loads YAML configs and merges them (run-specific config inherits from base).
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def load_config(config_path: str, base_path: str = "configs/base.yaml") -> dict:
    """Load a run-specific config and merge with base config."""
    project_root = Path(__file__).parent.parent.parent

    # Load base config
    base_file = project_root / base_path
    with open(base_file, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Load run-specific config
    run_file = project_root / config_path
    with open(run_file, "r") as f:
        run_cfg = yaml.safe_load(f)

    # Deep merge: run_cfg overrides base_cfg
    merged = _deep_merge(base_cfg, run_cfg)

    # Resolve paths relative to project root
    merged["_project_root"] = str(project_root)

    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts; override takes precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_device():
    """Get the best available device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
