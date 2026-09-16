from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_mlp(cfg: dict, num_classes: int) -> MLP:
    model_cfg = cfg.get("model", {}).get("mlp", {})

    return MLP(
        num_classes=num_classes,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.3),
    )