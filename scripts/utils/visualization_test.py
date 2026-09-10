import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from utils.visualization import (
    plot_confusion_matrix,
    plot_curves,
    plot_model_comparison,
)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # --------------------------------------------------
    # 1. Training curves
    # --------------------------------------------------
    history = [
        {
            "epoch": 1,
            "train_loss": 1.2,
            "val_loss": 1.0,
            "train_accuracy": 0.50,
            "val_accuracy": 0.55,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "val_loss": 0.7,
            "train_accuracy": 0.70,
            "val_accuracy": 0.75,
        },
        {
            "epoch": 3,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "train_accuracy": 0.85,
            "val_accuracy": 0.80,
        },
    ]

    curve_path = tmp / "curves.png"

    plot_curves(
        history,
        save_path=curve_path,
    )

    assert curve_path.exists()
    assert curve_path.stat().st_size > 0


    # --------------------------------------------------
    # 2. Curves with empty history
    # --------------------------------------------------
    empty_curve = tmp / "empty_curves.png"

    plot_curves(
        [],
        save_path=empty_curve,
    )

    # Empty history should return without creating a figure
    assert not empty_curve.exists()


    # --------------------------------------------------
    # 3. Confusion matrix
    # --------------------------------------------------
    cm = np.array([
        [8, 1, 1],
        [2, 7, 1],
        [0, 1, 9],
    ])

    cm_path = tmp / "cm.png"

    plot_confusion_matrix(
        cm,
        class_names=["A", "B", "C"],
        save_path=cm_path,
    )

    assert cm_path.exists()
    assert cm_path.stat().st_size > 0


    # --------------------------------------------------
    # 4. Normalized confusion matrix
    # --------------------------------------------------
    normalized_cm_path = tmp / "cm_normalized.png"

    plot_confusion_matrix(
        cm,
        class_names=["A", "B", "C"],
        save_path=normalized_cm_path,
        normalize=True,
    )

    assert normalized_cm_path.exists()
    assert normalized_cm_path.stat().st_size > 0


    # --------------------------------------------------
    # 5. 102-class confusion matrix
    # --------------------------------------------------
    cm_102 = np.zeros((102, 102), dtype=int)
    np.fill_diagonal(cm_102, 1)

    cm_102_path = tmp / "cm_102.png"

    plot_confusion_matrix(
        cm_102,
        class_names=[str(i) for i in range(102)],
        save_path=cm_102_path,
    )

    assert cm_102_path.exists()
    assert cm_102_path.stat().st_size > 0


    # --------------------------------------------------
    # 6. Model comparison
    # --------------------------------------------------
    results = {
        "MLP": {"accuracy": 0.70},
        "CNN": {"accuracy": 0.82},
        "ViT": {"accuracy": 0.88},
    }

    comparison_path = tmp / "comparison.png"

    plot_model_comparison(
        results,
        metric="accuracy",
        save_path=comparison_path,
    )

    assert comparison_path.exists()
    assert comparison_path.stat().st_size > 0


    # --------------------------------------------------
    # 7. Empty model comparison
    # --------------------------------------------------
    empty_comparison_path = tmp / "empty_comparison.png"

    plot_model_comparison(
        {},
        save_path=empty_comparison_path,
    )

    assert empty_comparison_path.exists()
    assert empty_comparison_path.stat().st_size > 0


    # --------------------------------------------------
    # 8. Missing metric falls back to 0
    # --------------------------------------------------
    missing_metric_path = tmp / "missing_metric.png"

    plot_model_comparison(
        {
            "CNN": {"accuracy": 0.8},
            "ViT": {},
        },
        metric="macro_f1",
        save_path=missing_metric_path,
    )

    assert missing_metric_path.exists()
    assert missing_metric_path.stat().st_size > 0