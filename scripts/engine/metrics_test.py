import numpy as np
import torch

from engine.metrics import MetricTracker


# --------------------------------------------------
# 1. Basic metric computation
# --------------------------------------------------
tracker = MetricTracker(num_classes=3, average="macro")

logits = torch.tensor([
    [5.0, 1.0, 0.0],  # -> 0, target 0
    [0.0, 5.0, 1.0],  # -> 1, target 1
    [0.0, 1.0, 5.0],  # -> 2, target 2
    [5.0, 1.0, 0.0],  # -> 0, target 1 (wrong)
])

targets = torch.tensor([0, 1, 2, 1])

tracker.update(logits, targets, loss=0.5)

metrics = tracker.compute()

assert metrics["accuracy"] == 0.75
assert 0.0 <= metrics["precision"] <= 1.0
assert 0.0 <= metrics["recall"] <= 1.0
assert 0.0 <= metrics["f1"] <= 1.0
assert 0.0 <= metrics["macro_f1"] <= 1.0
assert 0.0 <= metrics["weighted_f1"] <= 1.0
assert metrics["loss"] == 0.5


# --------------------------------------------------
# 2. Confusion matrix
# --------------------------------------------------
cm = tracker.confusion()

assert isinstance(cm, np.ndarray)
assert cm.shape == (3, 3)
assert cm.sum() == 4

expected_cm = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
])

assert np.array_equal(cm, expected_cm)


# --------------------------------------------------
# 3. Classification report
# --------------------------------------------------
report = tracker.report(
    class_names=["class_0", "class_1", "class_2"]
)

assert isinstance(report, str)
assert "class_0" in report
assert "class_1" in report
assert "class_2" in report
assert "accuracy" in report


# --------------------------------------------------
# 4. Reset
# --------------------------------------------------
tracker.reset()

assert tracker.y_true == []
assert tracker.y_pred == []
assert tracker.total_loss == 0.0
assert tracker.n_samples == 0
assert tracker.compute() == {}


# --------------------------------------------------
# 5. Loss accumulation across different batch sizes
# --------------------------------------------------
tracker.update(
    torch.tensor([
        [5.0, 0.0],
        [0.0, 5.0],
    ]),
    torch.tensor([0, 1]),
    loss=1.0,
)

tracker.update(
    torch.tensor([
        [5.0, 0.0],
    ]),
    torch.tensor([0]),
    loss=3.0,
)

metrics = tracker.compute()

assert tracker.n_samples == 3
assert metrics["loss"] == (1.0 * 2 + 3.0 * 1) / 3


# --------------------------------------------------
# 6. Fixed confusion matrix size
# --------------------------------------------------
tracker.reset()

tracker.update(
    torch.tensor([
        [5.0, 0.0, 0.0],
    ]),
    torch.tensor([0]),
)

cm = tracker.confusion()

assert cm.shape == (3, 3)
assert cm.sum() == 1
assert cm[0, 0] == 1
assert np.count_nonzero(cm) == 1