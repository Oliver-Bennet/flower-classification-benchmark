import torch

from utils.config import load_config
from models.convnext import build_convnext

cfg = load_config("configs/config.yaml")

model = build_convnext(cfg, num_classes=cfg["data"]["num_classes"])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"\nTotal parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)

assert y.shape == (2, 102)
assert total_params > 0
assert trainable_params == total_params