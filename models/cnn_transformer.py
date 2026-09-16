from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CNNStem(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        channels: Optional[List[int]] = None,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 128]

        layers: List[nn.Module] = []
        prev = in_channels

        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        prev,
                        c,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
            )
            prev = c

        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) 


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.norm1(x)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            need_weights=False,
        )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x


class CNNTransformer(nn.Module):

    def __init__(
        self,
        num_classes: int = 102,
        in_channels: int = 3,
        cnn_channels: Optional[List[int]] = None,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128, 128]

        self.stem = CNNStem(
            in_channels=in_channels,
            channels=cnn_channels,
        )

        feat_h = image_size // 8
        feat_w = image_size // 8

        self.num_tokens = feat_h * feat_w

        self.proj = nn.Linear(
            self.stem.out_channels,
            embed_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens + 1,
                embed_dim,
            )
        )

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        feat = self.stem(x)

        feat = feat.flatten(2).transpose(1, 2)

        tokens = self.proj(feat)

        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat(
            [cls, tokens],
            dim=1,
        )

        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]

        return self.head(cls_out)


def build_cnn_transformer(
    cfg: dict,
    num_classes: int,
) -> CNNTransformer:
    data = cfg.get("data", {})
    m = cfg.get("model", {}).get("cnn_transformer", {})

    return CNNTransformer(
        num_classes=num_classes,
        in_channels=3,
        cnn_channels=m.get(
            "cnn_channels",
            [64, 128, 128],
        ),
        embed_dim=m.get("embed_dim", 256),
        num_heads=m.get("num_heads", 4),
        num_layers=m.get("num_layers", 2),
        mlp_ratio=m.get("mlp_ratio", 4.0),
        dropout=m.get("dropout", 0.1),
        image_size=data.get("image_size", 224),
    )
