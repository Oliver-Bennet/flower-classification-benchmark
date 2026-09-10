import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from engine.inference import load_image, predict


# --------------------------------------------------
# Deterministic model
# --------------------------------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(
            torch.tensor([1.0, 3.0, 2.0])
        )

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.size(0), -1)


model = DummyModel()
device = torch.device("cpu")

class_names = ["class_0", "class_1", "class_2"]


# --------------------------------------------------
# 1. Tensor [C,H,W]
# --------------------------------------------------
tensor_image = torch.randn(3, 8, 8)

results = predict(
    model,
    tensor_image,
    device=device,
    class_names=class_names,
    top_k=2,
)

assert isinstance(results, list)
assert len(results) == 2

# Highest logit is class_1, then class_2
assert results[0][0] == "class_1"
assert results[1][0] == "class_2"

assert results[0][1] >= results[1][1]
assert all(0.0 <= prob <= 1.0 for _, prob in results)


# --------------------------------------------------
# 2. Probabilities sum to 1
# --------------------------------------------------
all_results = predict(
    model,
    tensor_image,
    device=device,
    class_names=class_names,
    top_k=3,
)

assert len(all_results) == 3

prob_sum = sum(prob for _, prob in all_results)

assert abs(prob_sum - 1.0) < 1e-6


# --------------------------------------------------
# 3. Tensor [B,C,H,W]
# --------------------------------------------------
batch = torch.randn(4, 3, 8, 8)

results = predict(
    model,
    batch,
    device=device,
    class_names=class_names,
    top_k=3,
)

# predict() intentionally returns predictions for first sample
assert len(results) == 3
assert results[0][0] == "class_1"


# --------------------------------------------------
# 4. PIL image requires transform
# --------------------------------------------------
pil_image = Image.new("RGB", (8, 8), (255, 0, 0))

try:
    predict(
        model,
        pil_image,
        device=device,
        class_names=class_names,
    )
except ValueError:
    pass
else:
    raise AssertionError(
        "PIL input without transform must raise ValueError"
    )


# --------------------------------------------------
# 5. PIL image with transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

results = predict(
    model,
    pil_image,
    transform=transform,
    device=device,
    class_names=class_names,
    top_k=2,
)

assert len(results) == 2
assert results[0][0] == "class_1"


# --------------------------------------------------
# 6. Invalid top_k
# --------------------------------------------------
for invalid_k in (0, -1):
    try:
        predict(
            model,
            tensor_image,
            device=device,
            top_k=invalid_k,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "top_k <= 0 must raise ValueError"
        )


# --------------------------------------------------
# 7. Invalid tensor shape
# --------------------------------------------------
try:
    predict(
        model,
        torch.randn(8, 8),
        device=device,
    )
except ValueError:
    pass
else:
    raise AssertionError(
        "Invalid tensor shape must raise ValueError"
    )


# --------------------------------------------------
# 8. load_image
# --------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "test.png"

    original = Image.new(
        "RGBA",
        (10, 12),
        (10, 20, 30, 255),
    )
    original.save(path)

    loaded = load_image(path)

    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"
    assert loaded.size == (10, 12)


# --------------------------------------------------
# 9. Model remains in eval mode
# --------------------------------------------------
assert model.training is False