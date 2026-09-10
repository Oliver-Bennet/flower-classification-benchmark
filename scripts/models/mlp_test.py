from utils.config import load_config
from models.mlp import build_mlp
import torch

cfg = load_config("configs/config.yaml")

model = build_mlp(
    cfg,
    num_classes=cfg["data"]["num_classes"],
)

print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nTotal parameters    :", total_params)
print("Trainable parameters:", trainable_params)

x = torch.randn(2, 3, 224, 224)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)