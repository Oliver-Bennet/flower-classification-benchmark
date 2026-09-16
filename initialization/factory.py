from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn


def _xavier_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _kaiming_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_initialization(model: nn.Module, name: str = "default") -> nn.Module:

    name = name.lower()
    if name == "default":
        return model
    if name in ("xavier", "glorot"):
        model.apply(_xavier_init)
        return model
    if name in ("kaiming", "he"):
        model.apply(_kaiming_init)
        return model
    raise ValueError(f"Unknown initialization: {name}")


def apply_from_config(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    name = (
        cfg.get("training", {})
        .get("initialization", {})
        .get("name", "default")
    )
    return apply_initialization(model, name)