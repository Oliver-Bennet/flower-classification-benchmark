"""
Optimizer factory.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch.optim import Optimizer


def build_optimizer(
    model_params: Iterable,
    cfg: Dict[str, Any],
) -> Optimizer:
    """
    Build optimizer from config['training']['optimizer'].
    """
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(
            model_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(
            model_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")