import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.evaluator import Evaluator


torch.manual_seed(42)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
x = torch.randn(10, 3, 8, 8)
y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

loader = DataLoader(
    TensorDataset(x, y),
    batch_size=2,
    shuffle=False,
)


# --------------------------------------------------
# Real model
# --------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 3),
        )

    def forward(self, x):
        return self.net(x)


model = TinyModel()

evaluator = Evaluator(
    model=model,
    device=torch.device("cpu"),
    num_classes=3,
    average="macro",
    use_amp=False,
)


# --------------------------------------------------
# 1. Evaluate without criterion
# --------------------------------------------------
result = evaluator.evaluate(
    loader,
    criterion=None,
    class_names=["class_0", "class_1", "class_2"],
)

assert isinstance(result, dict)

assert "metrics" in result
assert "confusion_matrix" in result
assert "classification_report" in result

metrics = result["metrics"]

assert metrics["num_samples"] == 10
assert metrics["inference_time_sec"] >= 0.0
assert metrics["inference_time_per_image_ms"] >= 0.0

assert 0.0 <= metrics["accuracy"] <= 1.0
assert 0.0 <= metrics["macro_f1"] <= 1.0
assert 0.0 <= metrics["weighted_f1"] <= 1.0

# No criterion => loss should not be present
assert "loss" not in metrics


# --------------------------------------------------
# 2. Confusion matrix
# --------------------------------------------------
cm = result["confusion_matrix"]

assert isinstance(cm, np.ndarray)
assert cm.shape == (3, 3)
assert cm.sum() == 10


# --------------------------------------------------
# 3. Classification report
# --------------------------------------------------
report = result["classification_report"]

assert isinstance(report, str)
assert "class_0" in report
assert "class_1" in report
assert "class_2" in report
assert "accuracy" in report


# --------------------------------------------------
# 4. Evaluate with criterion
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()

result_with_loss = evaluator.evaluate(
    loader,
    criterion=criterion,
)

metrics_with_loss = result_with_loss["metrics"]

assert metrics_with_loss["num_samples"] == 10
assert "loss" in metrics_with_loss
assert metrics_with_loss["loss"] >= 0.0


# --------------------------------------------------
# 5. Model must be in eval mode
# --------------------------------------------------
assert model.training is False


# --------------------------------------------------
# 6. AMP disabled on CPU
# --------------------------------------------------
assert evaluator.use_amp is False