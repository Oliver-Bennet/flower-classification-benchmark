import torch
import torch.nn as nn

from initialization.factory import (
    apply_initialization,
    apply_from_config,
)

# ============================================================
# 1. DEFAULT: weights must remain unchanged
# ============================================================
model = nn.Linear(100, 10)

before_w = model.weight.detach().clone()
before_b = model.bias.detach().clone()

returned = apply_initialization(model, "default")

assert returned is model
assert torch.equal(model.weight, before_w), "Default changed weights"
assert torch.equal(model.bias, before_b), "Default changed bias"


# ============================================================
# 2. XAVIER: verify actual initialization statistics
# ============================================================
torch.manual_seed(42)

model = nn.Linear(100, 10)
apply_initialization(model, "xavier")

w = model.weight.detach()

expected_std = (2.0 / (100 + 10)) ** 0.5

# Xavier uniform has bounded range
expected_bound = (6.0 / (100 + 10)) ** 0.5

assert w.max() <= expected_bound + 1e-6
assert w.min() >= -expected_bound - 1e-6
assert abs(w.mean().item()) < 0.05

assert torch.all(model.bias == 0), "Xavier bias is not zero"


# ============================================================
# 3. KAIMING: verify actual initialization statistics
# ============================================================
torch.manual_seed(42)

model = nn.Linear(100, 10)
apply_initialization(model, "kaiming")

w = model.weight.detach()

# fan_in = 100, Kaiming normal with ReLU
expected_std = (2.0 / 100) ** 0.5

actual_std = w.std().item()

assert abs(actual_std - expected_std) < 0.03, (
    f"Kaiming std incorrect: {actual_std:.4f}, "
    f"expected around {expected_std:.4f}"
)

assert torch.all(model.bias == 0), "Kaiming bias is not zero"


# ============================================================
# 4. CONFIG: must actually read training.initialization.name
# ============================================================
cfg = {
    "training": {
        "initialization": {
            "name": "xavier"
        }
    }
}

torch.manual_seed(42)

model = nn.Linear(100, 10)
apply_from_config(model, cfg)

w = model.weight.detach()

assert w.max() <= expected_bound + 1e-6
assert w.min() >= -expected_bound - 1e-6
assert torch.all(model.bias == 0)


# ============================================================
# 5. INVALID NAME: must raise ValueError
# ============================================================
try:
    apply_initialization(nn.Linear(10, 2), "invalid_method")
    raise AssertionError("Expected ValueError was not raised")
except ValueError as e:
    assert "Unknown initialization" in str(e)
