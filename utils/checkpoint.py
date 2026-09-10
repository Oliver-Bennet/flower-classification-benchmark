"""
Checkpoint save / load utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "config": config,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(extra)

    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(state["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state