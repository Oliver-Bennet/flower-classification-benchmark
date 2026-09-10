import torch
import torch.nn as nn

from schedulers.factory import build_scheduler


def make_optimizer():
    model = nn.Linear(10, 3)
    return torch.optim.SGD(model.parameters(), lr=0.1)


# ============================================================
# 1. NONE
# ============================================================
cfg = {
    "training": {
        "scheduler": {
            "name": "none"
        }
    }
}

optimizer = make_optimizer()
scheduler = build_scheduler(optimizer, cfg)

assert scheduler is None


# ============================================================
# 2. StepLR
# ============================================================
cfg = {
    "training": {
        "epochs": 30,
        "scheduler": {
            "name": "step",
            "step_size": 2,
            "gamma": 0.1,
        }
    }
}

optimizer = make_optimizer()
scheduler = build_scheduler(optimizer, cfg)

assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
assert scheduler.step_size == 2
assert scheduler.gamma == 0.1

lr_before = optimizer.param_groups[0]["lr"]

# Simulate 2 epochs
optimizer.step()
scheduler.step()

optimizer.step()
scheduler.step()

lr_after = optimizer.param_groups[0]["lr"]

assert lr_before == 0.1
assert abs(lr_after - 0.01) < 1e-10, (
    f"StepLR did not reduce LR correctly: {lr_after}"
)


# ============================================================
# 3. CosineAnnealingLR
# ============================================================
cfg = {
    "training": {
        "epochs": 4,
        "scheduler": {
            "name": "cosine",
            "T_max": 4,
            "eta_min": 0.001,
        }
    }
}

optimizer = make_optimizer()
scheduler = build_scheduler(optimizer, cfg)

assert isinstance(
    scheduler,
    torch.optim.lr_scheduler.CosineAnnealingLR
)
assert scheduler.T_max == 4
assert scheduler.eta_min == 0.001

lrs = [optimizer.param_groups[0]["lr"]]

for _ in range(4):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

assert lrs[0] == 0.1
assert abs(lrs[-1] - 0.001) < 1e-10

# LR should decrease monotonically for this setup
assert all(
    lrs[i + 1] <= lrs[i]
    for i in range(len(lrs) - 1)
), f"Cosine LR is not decreasing: {lrs}"


# ============================================================
# 4. OneCycleLR
# ============================================================
cfg = {
    "training": {
        "epochs": 2,
        "scheduler": {
            "name": "onecycle",
            "max_lr": 0.1,
            "pct_start": 0.3,
            "div_factor": 10.0,
            "final_div_factor": 100.0,
        }
    }
}

steps_per_epoch = 5

optimizer = make_optimizer()
scheduler = build_scheduler(
    optimizer,
    cfg,
    steps_per_epoch=steps_per_epoch,
)

assert isinstance(
    scheduler,
    torch.optim.lr_scheduler.OneCycleLR
)

# OneCycleLR has total_steps = epochs * steps_per_epoch
assert scheduler.total_steps == 10

lrs = []

for _ in range(10):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

# OneCycle should change LR
assert len(set(lrs)) > 2, (
    f"OneCycleLR did not change LR correctly: {lrs}"
)

# max_lr should be reached approximately
assert max(lrs) <= 0.1 + 1e-6


# ============================================================
# 5. OneCycleLR WITHOUT steps_per_epoch -> ValueError
# ============================================================
cfg = {
    "training": {
        "epochs": 2,
        "scheduler": {
            "name": "onecycle",
            "max_lr": 0.1,
        }
    }
}

optimizer = make_optimizer()

try:
    build_scheduler(optimizer, cfg)
except ValueError as e:
    assert "steps_per_epoch" in str(e)
else:
    raise AssertionError(
        "OneCycleLR should require steps_per_epoch"
    )


# ============================================================
# 6. Unknown scheduler -> ValueError
# ============================================================
cfg = {
    "training": {
        "scheduler": {
            "name": "invalid_scheduler"
        }
    }
}

optimizer = make_optimizer()

try:
    build_scheduler(optimizer, cfg)
except ValueError as e:
    assert "Unknown scheduler" in str(e)
else:
    raise AssertionError(
        "Unknown scheduler should raise ValueError"
    )


