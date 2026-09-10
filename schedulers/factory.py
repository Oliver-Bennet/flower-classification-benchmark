"""
Learning-rate scheduler factory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: Optional[int] = None,
) -> Optional[_LRScheduler]:
    """
    Build LR scheduler from config['training']['scheduler'].

    For OneCycleLR, steps_per_epoch must be provided (len(train_loader)).
    """
    sch_cfg = cfg.get("training", {}).get("scheduler", {})
    name = sch_cfg.get("name", "cosine").lower()

    if name in ("none", "null", ""):
        return None

    epochs = cfg.get("training", {}).get("epochs", 30)

    if name == "step":
        return StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )

    if name in ("cosine", "cosineannealing", "cosine_annealing"):
        T_max = int(sch_cfg.get("T_max", epochs))
        eta_min = float(sch_cfg.get("eta_min", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if name in ("onecycle", "one_cycle"):
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR requires steps_per_epoch")
        max_lr = float(sch_cfg.get("max_lr", 1e-2))
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=float(sch_cfg.get("pct_start", 0.3)),
            div_factor=float(sch_cfg.get("div_factor", 25.0)),
            final_div_factor=float(sch_cfg.get("final_div_factor", 1e4)),
        )

    raise ValueError(f"Unknown scheduler: {name}")