from __future__ import annotations

import torch.nn as nn


def build_swin(cfg: dict, num_classes: int) -> nn.Module:

    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for Swin Transformer. Install with: pip install timm"
        ) from e

    m = cfg.get("model", {}).get("swin", {})
    variant = m.get("variant", "swin_tiny_patch4_window7_224")

    name_map = {
        "swin_t": "swin_tiny_patch4_window7_224",
        "swin_s": "swin_small_patch4_window7_224",
        "swin_b": "swin_base_patch4_window7_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
    }
    model_name = name_map.get(variant, variant)

    model = timm.create_model(
        model_name,
        pretrained=False,         
        num_classes=num_classes,
    )
    return model