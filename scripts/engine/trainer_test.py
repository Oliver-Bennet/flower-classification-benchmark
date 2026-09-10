import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.trainer import Trainer
from optimizers.factory import build_optimizer
from schedulers.factory import build_scheduler


# --------------------------------------------------
# Small deterministic dataset
# --------------------------------------------------
torch.manual_seed(42)

x_train = torch.randn(16, 3, 8, 8)
y_train = torch.randint(0, 2, (16,))

x_val = torch.randn(8, 3, 8, 8)
y_val = torch.randint(0, 2, (8,))

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=4,
    shuffle=False,
)

val_loader = DataLoader(
    TensorDataset(x_val, y_val),
    batch_size=4,
    shuffle=False,
)


# --------------------------------------------------
# Tiny real model
# --------------------------------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 8 * 8, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,
    gamma=0.5,
)


# --------------------------------------------------
# Temporary output directories
# --------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    cfg = {
        "training": {
            "epochs": 2,
            "amp": False,
            "gradient_clip": 1.0,
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.0,
                "monitor": "val_macro_f1",
                "mode": "max",
            },
        },
        "evaluation": {
            "average": "macro",
        },
        "logging": {
            "log_dir": str(tmp / "logs"),
            "checkpoint_dir": str(tmp / "checkpoints"),
        },
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        scheduler=scheduler,
        cfg=cfg,
        experiment_name="trainer_test",
        num_classes=2,
    )

    # Check initial state
    assert trainer.epochs == 2
    assert trainer.num_classes == 2
    assert trainer.early_stopping is not None
    assert trainer.scheduler_per_batch is False

    # Save initial parameters
    initial_state = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }

    result = trainer.fit(train_loader, val_loader)

    # --------------------------------------------------
    # Result structure
    # --------------------------------------------------
    assert isinstance(result, dict)

    assert "best_metric" in result
    assert "best_checkpoint" in result
    assert "total_training_time" in result
    assert "history" in result

    assert isinstance(result["history"], list)
    assert len(result["history"]) == 2

    # --------------------------------------------------
    # Training actually changed model parameters
    # --------------------------------------------------
    final_state = model.state_dict()

    changed = any(
        not torch.equal(initial_state[k], final_state[k])
        for k in initial_state
    )

    assert changed

    # --------------------------------------------------
    # Check history contents
    # --------------------------------------------------
    for row in result["history"]:
        assert "train_loss" in row
        assert "train_accuracy" in row
        assert "val_loss" in row
        assert "val_accuracy" in row
        assert "val_macro_f1" in row
        assert "val_weighted_f1" in row
        assert "lr" in row
        assert "epoch_time" in row

        assert 0.0 <= row["train_accuracy"] <= 1.0
        assert 0.0 <= row["val_accuracy"] <= 1.0
        assert 0.0 <= row["val_macro_f1"] <= 1.0
        assert row["train_loss"] >= 0.0
        assert row["val_loss"] >= 0.0
        assert row["lr"] > 0.0
        assert row["epoch_time"] >= 0.0

    # --------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------
    checkpoint_path = Path(result["best_checkpoint"])

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert "epoch" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert "metrics" in checkpoint
    assert checkpoint["epoch"] in (1, 2)

    # --------------------------------------------------
    # Logger output
    # --------------------------------------------------
    csv_path = tmp / "logs" / "trainer_test_history.csv"
    json_path = tmp / "logs" / "trainer_test_history.json"

    assert csv_path.exists()
    assert json_path.exists()

    # --------------------------------------------------
    # Scheduler stepped once per epoch
    # --------------------------------------------------
    assert scheduler.last_epoch == 2
