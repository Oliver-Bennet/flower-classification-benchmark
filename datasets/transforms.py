"""
Image transforms for training / validation / test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torchvision.transforms as T


def get_mean_std(cfg: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Return (mean, std) from config."""
    data = cfg.get("data", {})
    mean = data.get("mean", [0.485, 0.456, 0.406])
    std = data.get("std", [0.229, 0.224, 0.225])
    return mean, std


def build_train_transforms(cfg: Dict[str, Any]) -> T.Compose:
    """
    Build training transforms with optional data augmentation.
    """
    data = cfg.get("data", {})
    aug = data.get("augmentation", {})
    image_size = data.get("image_size", 224)
    mean, std = get_mean_std(cfg)

    transforms_list = [
        T.Resize((image_size, image_size)),
    ]

    if aug.get("horizontal_flip", True):
        transforms_list.append(
            T.RandomHorizontalFlip(p=aug.get("horizontal_flip_p", 0.5))
        )

    rotation = aug.get("rotation_degrees", 0)
    if rotation and rotation > 0:
        transforms_list.append(T.RandomRotation(degrees=rotation))

    if aug.get("color_jitter", False):
        transforms_list.append(
            T.ColorJitter(
                brightness=aug.get("color_jitter_brightness", 0.2),
                contrast=aug.get("color_jitter_contrast", 0.2),
                saturation=aug.get("color_jitter_saturation", 0.2),
                hue=aug.get("color_jitter_hue", 0.1),
            )
        )

    transforms_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    return T.Compose(transforms_list)


def build_eval_transforms(cfg: Dict[str, Any]) -> T.Compose:
    """
    Build deterministic transforms for validation and test.
    No random augmentation.
    """
    data = cfg.get("data", {})
    image_size = data.get("image_size", 224)
    mean, std = get_mean_std(cfg)

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def build_transforms(cfg: Dict[str, Any], split: str = "train") -> T.Compose:
    """Convenience wrapper."""
    if split.lower() in ("train", "training"):
        return build_train_transforms(cfg)
    return build_eval_transforms(cfg)