"""
Miscellaneous helper utilities for IMUSA.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(log_dir: str = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """Set up rich-formatted logging."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("imusa")
    logger.info(f"Logging to {log_file}")
    return logger


def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """Load JSON file to dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model) -> dict:
    """Count model parameters (total and trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_m": f"{total / 1e6:.1f}M",
        "trainable_m": f"{trainable / 1e6:.1f}M",
    }


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
