from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from datasets.flower_dataset import create_dataloaders, create_datasets
from engine.evaluator import Evaluator
from models import build_model, count_parameters
from utils.checkpoint import load_checkpoint
from utils.config import ensure_dirs, get_device, load_config
from utils.seed import set_seed
from utils.visualization import plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on test set")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--experiments", type=str, default="configs/experiments.yaml")
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.experiment, args.experiments)
    ensure_dirs(cfg)
    seed = cfg.get("project", {}).get("seed", 42)
    deterministic = cfg.get("project", {}).get("deterministic", False)
    set_seed(seed, deterministic=deterministic)
    device = get_device(cfg)

    train_ds, val_ds, test_ds = create_datasets(cfg)
    _, _, test_loader = create_dataloaders(cfg, train_ds, val_ds, test_ds)
    num_classes = train_ds.num_classes
    class_names = train_ds.classes

    model = build_model(cfg, num_classes)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    n_params = count_parameters(model)
    print(f"Parameters : {n_params:,}")
    print(f"Checkpoint : {args.checkpoint}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("training", {}).get("label_smoothing", 0.0)
    )
    evaluator = Evaluator(
        model,
        device,
        num_classes=num_classes,
        average=cfg.get("evaluation", {}).get("average", "macro"),
        use_amp=cfg.get("training", {}).get("amp", False),
    )
    result = evaluator.evaluate(test_loader, criterion, class_names)

    metrics = result["metrics"]
    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    print("\nClassification Report:")
    print(result["classification_report"])

    # Save
    exp_name = args.name or args.experiment

    if exp_name is None:
        exp_name = Path(args.checkpoint).stem

    exp_name = exp_name.replace(".", "_")
    result_dir = Path(cfg.get("logging", {}).get("result_dir", "outputs/results"))
    result_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "experiment": exp_name,
        "checkpoint": args.checkpoint,
        "num_parameters": n_params,
        "metrics": metrics,
    }
    with open(result_dir / f"{exp_name}_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    fig_dir = Path(cfg.get("logging", {}).get("figure_dir", "outputs/figures"))
    if cfg.get("evaluation", {}).get("save_confusion_matrix", True):
        plot_confusion_matrix(
            result["confusion_matrix"],
            class_names=class_names if num_classes <= 30 else None,
            save_path=fig_dir / f"{exp_name}_cm.png",
            title=f"{exp_name} – Confusion Matrix",
        )
    print("Done.")


if __name__ == "__main__":
    main()