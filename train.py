"""
Main training entry point.

Usage examples
--------------
# Train baseline CNN with default config
python train.py --experiment E2_cnn

# Train a specific architecture group experiment
python train.py --experiment architecture.E4_vit

# Override config path / output name
python train.py --experiment E2_cnn --name my_cnn_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.flower_dataset import create_dataloaders, create_datasets, print_dataset_summary
from engine.trainer import Trainer
from initialization import apply_from_config
from models import build_model, count_parameters
from optimizers import build_optimizer
from schedulers import build_scheduler
from utils.config import ensure_dirs, get_device, load_config
from utils.seed import set_seed
from utils.visualization import plot_curves

import torch
import torch.nn as nn

def parse_args():
    p = argparse.ArgumentParser(description="Flower Classification Benchmark – Train")
    p.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to base config.yaml",
    )
    p.add_argument(
        "--experiments",
        type=str,
        default="configs/experiments.yaml",
        help="Path to experiments.yaml",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (e.g. E2_cnn or architecture.E2_cnn)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override experiment name used for logs/checkpoints",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.experiment, args.experiments)
    ensure_dirs(cfg)

    exp_name = args.name or args.experiment or cfg.get("model", {}).get("name", "run")
    exp_name = exp_name.replace(".", "_")

    seed = cfg.get("project", {}).get("seed", 42)
    deterministic = cfg.get("project", {}).get("deterministic", False)

    set_seed(seed, deterministic=deterministic)

    device = get_device(cfg)
    print(f"Device          : {device}")
    print(f"Experiment      : {exp_name}")
    print(f"Model           : {cfg.get('model', {}).get('name')}")
    print(f"Optimizer       : {cfg.get('training', {}).get('optimizer', {}).get('name')}")
    print(f"Scheduler       : {cfg.get('training', {}).get('scheduler', {}).get('name')}")
    print(f"Initialization  : {cfg.get('training', {}).get('initialization', {}).get('name')}")

    # Data
    train_ds, val_ds, test_ds = create_datasets(cfg)
    print_dataset_summary(train_ds, val_ds, test_ds)
    train_loader, val_loader, _ = create_dataloaders(
        cfg, train_ds, val_ds, test_ds
    )

    num_classes = train_ds.num_classes
    cfg.setdefault("data", {})["num_classes"] = num_classes

    # Model
    model = build_model(cfg, num_classes)
    model = apply_from_config(model, cfg)
    n_params = count_parameters(model)
    print(f"Parameters      : {n_params:,}")

    # Optimizer / Scheduler / Loss
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(
        optimizer, cfg, steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("training", {}).get("label_smoothing", 0.0)
    )

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        cfg=cfg,
        experiment_name=exp_name,
        num_classes=num_classes,
    )
    result = trainer.fit(train_loader, val_loader)

    # Save summary
    summary = {
        "experiment": exp_name,
        "model": cfg.get("model", {}).get("name"),
        "num_parameters": n_params,
        "best_metric": result["best_metric"],
        "best_checkpoint": result["best_checkpoint"],
        "total_training_time_sec": result["total_training_time"],
        "device": str(device),
        "seed": seed,
        "monitor": result.get("monitor"),
        "best_epoch": result.get("best_epoch"),
        "num_classes": num_classes,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    result_dir = Path(cfg.get("logging", {}).get("result_dir", "outputs/results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / f"{exp_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")

    # Curves
    fig_dir = Path(cfg.get("logging", {}).get("figure_dir", "outputs/figures"))
    plot_curves(
        result["history"],
        save_path=fig_dir / f"{exp_name}_curves.png",
        title=f"{exp_name} – Training Curves",
    )
    print("Done.")


if __name__ == "__main__":
    main()