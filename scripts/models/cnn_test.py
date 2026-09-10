from utils.config import load_config
from models.cnn import build_cnn
import torch

cfg = load_config("configs/config.yaml")

model = build_cnn(cfg, num_classes=102)

print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters    :", total_params)
print("Trainable parameters:", trainable_params)

x = torch.randn(2, 3, 224, 224)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)