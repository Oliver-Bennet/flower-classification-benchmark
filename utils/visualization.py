from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_curves(
    history: List[Dict[str, Any]],
    save_path: Optional[str | Path] = None,
    title: str = "Training Curves",
) -> None:
    if not history:
        return

    epochs = [h["epoch"] if "epoch" in h else i + 1 for i, h in enumerate(history)]
    train_loss = [h.get("train_loss") for h in history]
    val_loss = [h.get("val_loss") for h in history]
    train_acc = [h.get("train_accuracy") for h in history]
    val_acc = [h.get("val_accuracy") for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False if cm.shape[0] > 20 else True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names if class_names and len(class_names) <= 30 else False,
        yticklabels=class_names if class_names and len(class_names) <= 30 else False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    save_path: Optional[str | Path] = None,
    title: str = "Model Comparison",
) -> None:
    names = list(results.keys())
    values = [results[n].get(metric, 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color="steelblue")
    ax.set_ylabel(metric)
    ax.set_title(title)
    y_max = max(values) if values else 0.0
    ax.set_ylim(0, max(y_max * 1.15, 1.0))
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)