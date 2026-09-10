import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from utils.checkpoint import save_checkpoint, load_checkpoint


# ============================================================
# 1. Create model / optimizer / scheduler
# ============================================================
model = nn.Linear(10, 3)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.1,
)

# Create optimizer state
x = torch.randn(8, 10)
target = torch.randint(0, 3, (8,))

loss = nn.CrossEntropyLoss()(model(x), target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()


# ============================================================
# 2. Save checkpoint
# ============================================================
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "checkpoint.pt"

    metrics = {
        "val_accuracy": 0.85,
        "val_macro_f1": 0.82,
    }

    config = {
        "training": {
            "epochs": 30,
        }
    }

    extra = {
        "experiment": "test_checkpoint",
    }

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=7,
        metrics=metrics,
        config=config,
        extra=extra,
    )

    assert path.exists(), "Checkpoint file was not created"


    # ========================================================
    # 3. Create fresh objects
    # ========================================================
    new_model = nn.Linear(10, 3)

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    new_scheduler = torch.optim.lr_scheduler.StepLR(
        new_optimizer,
        step_size=2,
        gamma=0.1,
    )


    # ========================================================
    # 4. Load checkpoint
    # ========================================================
    state = load_checkpoint(
        path=path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        device=torch.device("cpu"),
    )


    # ========================================================
    # 5. Check returned metadata
    # ========================================================
    assert state["epoch"] == 7
    assert state["metrics"] == metrics
    assert state["config"] == config
    assert state["experiment"] == "test_checkpoint"


    # ========================================================
    # 6. Check model weights
    # ========================================================
    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            new_model.state_dict()[key],
        ), f"Model parameter mismatch: {key}"


    # ========================================================
    # 7. Check optimizer state
    # ========================================================
    assert (
        new_optimizer.state_dict()["param_groups"]
        == optimizer.state_dict()["param_groups"]
    )

    assert (
        new_optimizer.state_dict()["state"].keys()
        == optimizer.state_dict()["state"].keys()
    )


    # ========================================================
    # 8. Check scheduler state
    # ========================================================
    assert (
        new_scheduler.state_dict()
        == scheduler.state_dict()
    )


    # ========================================================
    # 9. Missing checkpoint must raise FileNotFoundError
    # ========================================================
    missing_path = Path(tmp) / "does_not_exist.pt"

    try:
        load_checkpoint(
            missing_path,
            nn.Linear(10, 3),
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Missing checkpoint should raise FileNotFoundError"
        )


