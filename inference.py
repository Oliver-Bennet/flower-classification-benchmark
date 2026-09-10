"""
Single-image inference CLI.

Usage
-----
python inference.py --checkpoint outputs/checkpoints/E2_cnn_best.pth \
                    --image path/to/flower.jpg \
                    --experiment E2_cnn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.flower_dataset import create_datasets
from datasets.transforms import build_eval_transforms
from engine.inference import load_image, predict
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import get_device, load_config
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Single image inference")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--experiments", type=str, default="configs/experiments.yaml")
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--top_k", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be greater than 0")
    cfg = load_config(args.config, args.experiment, args.experiments)

    seed = cfg.get("project", {}).get("seed", 42)
    deterministic = cfg.get("project", {}).get("deterministic", False)
    set_seed(seed, deterministic=deterministic)

    device = get_device(cfg)

    # Need class names from dataset
    train_ds, _, _ = create_datasets(cfg)
    class_names = train_ds.classes
    num_classes = train_ds.num_classes

    model = build_model(cfg, num_classes)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    transform = build_eval_transforms(cfg)
    
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    image = load_image(args.image)

    results = predict(
        model,
        image,
        transform=transform,
        device=device,
        class_names=class_names,
        top_k=args.top_k,
    )

    print(f"Image: {args.image}")
    print("Top predictions:")
    for rank, (name, prob) in enumerate(results, 1):
        print(f"  {rank}. {name:30s}  {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()