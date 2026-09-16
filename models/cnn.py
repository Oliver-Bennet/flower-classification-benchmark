from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 102,
        in_channels: int = 3,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers: List[nn.Module] = []
        prev = in_channels
        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, c, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(pool_size),
                ]
            )
            prev = c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def build_cnn(cfg: dict, num_classes: int) -> CNN:
    model_cfg = cfg.get("model", {}).get("cnn", {})
    return CNN(
        num_classes=num_classes,
        in_channels=3,
        channels=model_cfg.get("channels", [32, 64, 128, 256]),
        kernel_size=model_cfg.get("kernel_size", 3),
        pool_size=model_cfg.get("pool_size", 2),
        dropout=cfg.get("model", {}).get("dropout", 0.3),
    )