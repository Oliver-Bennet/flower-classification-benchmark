from __future__ import annotations

import torch.nn as nn


def build_convnext(cfg: dict, num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for ConvNeXt. Install with: pip install timm"
        ) from e

    m = cfg.get("model", {}).get("convnext", {})
    variant = m.get("variant", "convnext_tiny")

    name_map = {
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
        "convnext_t": "convnext_tiny",
        "convnext_s": "convnext_small",
    }
    model_name = name_map.get(variant, variant)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )
    return model