"""
Flower Classification Dataset + DataLoader factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from datasets.transforms import build_eval_transforms, build_train_transforms
from utils.seed import worker_init_fn


class FlowerDataset(ImageFolder):
    """
    Thin wrapper around torchvision.datasets.ImageFolder.
    Provides class_to_idx, classes, and a few convenience helpers.
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root=str(root),
            transform=transform,
            target_transform=target_transform,
        )

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_distribution(self) -> Dict[str, int]:
        """Return {class_name: count}."""
        from collections import Counter

        counts = Counter(self.targets)
        return {self.classes[i]: counts[i] for i in range(len(self.classes))}


def _resolve_split_dir(root: Path, split_name: str, alternatives: List[str]) -> Path:
    """Try primary name then alternatives (e.g. valid / val)."""
    candidates = [split_name] + alternatives
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find split directory under {root}. "
        f"Tried: {candidates}"
    )


def create_datasets(
    cfg: Dict[str, Any],
) -> Tuple[FlowerDataset, FlowerDataset, FlowerDataset]:
    """
    Create train / val / test FlowerDataset instances.

    Returns
    -------
    train_ds, val_ds, test_ds
    """
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/raw"))

    train_dir = _resolve_split_dir(
        root,
        data_cfg.get("train_dir", "train"),
        alternatives=[],
    )
    val_dir = _resolve_split_dir(
        root,
        data_cfg.get("val_dir", "valid"),
        alternatives=["val", "validation"],
    )
    test_dir = _resolve_split_dir(
        root,
        data_cfg.get("test_dir", "test"),
        alternatives=[],
    )

    train_tf = build_train_transforms(cfg)
    eval_tf = build_eval_transforms(cfg)

    train_ds = FlowerDataset(train_dir, transform=train_tf)
    val_ds = FlowerDataset(val_dir, transform=eval_tf)
    test_ds = FlowerDataset(test_dir, transform=eval_tf)

    # Sanity: same class set
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(
        "Train / Val class_to_idx mappings do not match."
        )

    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            "Train / Test class_to_idx mappings do not match."
        )

    return train_ds, val_ds, test_ds


def create_dataloaders(
    cfg: Dict[str, Any],
    train_ds: Optional[Dataset] = None,
    val_ds: Optional[Dataset] = None,
    test_ds: Optional[Dataset] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders. If datasets are not provided they are created from cfg.
    """
    if train_ds is None or val_ds is None or test_ds is None:
        train_ds, val_ds, test_ds = create_datasets(cfg)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", True)
    seed = cfg.get("project", {}).get("seed", 42)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=lambda wid: worker_init_fn(wid, seed),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def print_dataset_summary(
    train_ds: FlowerDataset,
    val_ds: FlowerDataset,
    test_ds: FlowerDataset,
) -> None:
    """Pretty-print basic statistics."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Number of classes : {train_ds.num_classes}")
    print(f"Classes (first 10): {train_ds.classes[:10]}")
    print(f"Train samples     : {len(train_ds)}")
    print(f"Val samples       : {len(val_ds)}")
    print(f"Test samples      : {len(test_ds)}")
    print(f"Total             : {len(train_ds) + len(val_ds) + len(test_ds)}")
    print("-" * 60)
    print("Class distribution (train):")
    dist = train_ds.class_distribution()
    for i, (name, cnt) in enumerate(sorted(dist.items(), key=lambda x: -x[1])):
        if i >= 10:
            print(f"  ... and {len(dist) - 10} more classes")
            break
        print(f"  {name:30s}: {cnt:4d}")
    print("=" * 60)