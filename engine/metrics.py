from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch


class MetricTracker:

    def __init__(self, num_classes: int, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.total_loss = 0.0
        self.n_samples = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        self.y_true.extend(targets_np.tolist())
        self.y_pred.extend(preds.tolist())

        batch_size = targets.size(0)
        self.n_samples += batch_size

        if loss is not None:
            self.total_loss += loss * batch_size

    def compute(self) -> Dict[str, float]:
        if not self.y_true:
            return {}

        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(
                    y_true, y_pred, average=self.average, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true, y_pred, average=self.average, zero_division=0
                )
            ),
            "f1": float(
                f1_score(y_true, y_pred, average=self.average, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }
        if self.n_samples > 0:
            metrics["loss"] = self.total_loss / self.n_samples
        return metrics

    def confusion(self) -> np.ndarray:
        return confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=list(range(self.num_classes)),
        )

    def report(self, class_names: Optional[List[str]] = None) -> str:
        """
        Return a detailed classification report.
        """
        return classification_report(
            self.y_true,
            self.y_pred,
            labels=list(range(self.num_classes)),
            target_names=class_names,
            zero_division=0,
        )