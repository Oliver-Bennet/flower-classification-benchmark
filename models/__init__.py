"""
Model factory: build any supported architecture from config.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from models.mlp import build_mlp
from models.cnn import build_cnn
from models.cnn_transformer import build_cnn_transformer
from models.vit import build_vit
from models.swin import build_swin
from models.convnext import build_convnext
from models.maxvit import build_maxvit


_BUILDERS = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "cnn_transformer": build_cnn_transformer,
    "vit": build_vit,
    "swin": build_swin,
    "convnext": build_convnext,
    "maxvit": build_maxvit,
}


def build_model(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    """
    Instantiate a model according to cfg['model']['name'].

    Always trains from scratch (pretrained=False for timm models).
    """
    name = cfg.get("model", {}).get("name", "cnn").lower()
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {list(_BUILDERS.keys())}"
        )
    model = _BUILDERS[name](cfg, num_classes)
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return number of parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

#$env:PYTHONPATH = (Get-Location).Path   