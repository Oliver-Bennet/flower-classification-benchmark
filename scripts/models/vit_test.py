import torch

from utils.config import load_config
from models.vit import build_vit


cfg = load_config("configs/config.yaml")

model = build_vit(cfg, num_classes=102)

print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print()
print("Total parameters    :", total_params)
print("Trainable parameters:", trainable_params)

x = torch.randn(2, 3, 224, 224)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)