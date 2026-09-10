"""
MaxViT via timm (pretrained=False → train from scratch).
"""

from __future__ import annotations

import torch.nn as nn


def build_maxvit(cfg: dict, num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for MaxViT. Install with: pip install timm"
        ) from e

    m = cfg.get("model", {}).get("maxvit", {})
    variant = m.get("variant", "maxvit_tiny_tf_224")

    name_map = {
        "maxvit_tiny": "maxvit_tiny_tf_224",
        "maxvit_small": "maxvit_small_tf_224",
        "maxvit_base": "maxvit_base_tf_224",
        "maxvit_t": "maxvit_tiny_tf_224",
    }
    model_name = name_map.get(variant, variant)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )
    return model