import torch
import torch.nn as nn

from optimizers.factory import build_optimizer


def make_model():
    return nn.Linear(10, 3)


# ============================================================
# 1. SGD
# ============================================================
cfg = {
    "training": {
        "optimizer": {
            "name": "sgd",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        }
    }
}

model = make_model()
optimizer = build_optimizer(model.parameters(), cfg)

assert isinstance(optimizer, torch.optim.SGD)
assert optimizer.defaults["lr"] == 0.01
assert optimizer.defaults["momentum"] == 0.9
assert optimizer.defaults["weight_decay"] == 0.0001


# ============================================================
# 2. Adam
# ============================================================
cfg = {
    "training": {
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
            "betas": [0.8, 0.95],
            "weight_decay": 0.0002,
        }
    }
}

model = make_model()
optimizer = build_optimizer(model.parameters(), cfg)

assert isinstance(optimizer, torch.optim.Adam)
assert optimizer.defaults["lr"] == 0.001
assert optimizer.defaults["betas"] == (0.8, 0.95)
assert optimizer.defaults["weight_decay"] == 0.0002


# ============================================================
# 3. AdamW
# ============================================================
cfg = {
    "training": {
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
        }
    }
}

model = make_model()
optimizer = build_optimizer(model.parameters(), cfg)

assert isinstance(optimizer, torch.optim.AdamW)
assert optimizer.defaults["lr"] == 0.001
assert optimizer.defaults["betas"] == (0.9, 0.999)
assert optimizer.defaults["weight_decay"] == 0.0001


# ============================================================
# 4. Optimizer must actually update model parameters
# ============================================================
model = make_model()

cfg = {
    "training": {
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0001,
        }
    }
}

optimizer = build_optimizer(model.parameters(), cfg)

x = torch.randn(8, 10)
target = torch.randint(0, 3, (8,))

before = model.weight.detach().clone()

loss = nn.CrossEntropyLoss()(model(x), target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

after = model.weight.detach()

assert not torch.equal(before, after), "Optimizer did not update parameters"


# ============================================================
# 5. Unknown optimizer must raise ValueError
# ============================================================
cfg = {
    "training": {
        "optimizer": {
            "name": "not_an_optimizer"
        }
    }
}

try:
    build_optimizer(make_model().parameters(), cfg)
except ValueError as e:
    assert "Unknown optimizer" in str(e)
else:
    raise AssertionError("Expected ValueError was not raised")


