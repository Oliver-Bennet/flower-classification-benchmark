from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.metrics import MetricTracker
from utils.timer import Timer


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 102,
        average: str = "macro",
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.average = average
        self.use_amp = use_amp and device.type == "cuda"

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.model.eval()
        tracker = MetricTracker(
            num_classes=self.num_classes,
            average=self.average,
        )

        timer = Timer()
        timer.start()

        for images, labels in tqdm(loader, desc="Evaluate", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(
            "cuda",
            enabled=self.use_amp,):
                logits = self.model(images)
                loss = None
                if criterion is not None:
                    loss = criterion(logits, labels).item()

            tracker.update(logits, labels, loss)

        elapsed = timer.stop()
        metrics = tracker.compute()
        metrics["inference_time_sec"] = elapsed
        metrics["inference_time_per_image_ms"] = (
            (elapsed / max(tracker.n_samples, 1)) * 1000.0
        )
        metrics["num_samples"] = tracker.n_samples

        result = {
            "metrics": metrics,
            "confusion_matrix": tracker.confusion(),
            "classification_report": tracker.report(class_names),
        }
        return result