from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def predict(
    model: nn.Module,
    image: Union[torch.Tensor, Image.Image],
    transform: Optional[transforms.Compose] = None,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
    top_k: int = 3,
) -> List[Tuple[str, float]]:

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if isinstance(image, Image.Image):
        if transform is None:
            raise ValueError(
                "transform is required when image is PIL.Image"
            )
        tensor = transform(image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() != 4:
            raise ValueError(
                "image tensor must have shape [C,H,W] or [B,C,H,W]"
            )
    else:
        raise TypeError(
            "image must be a torch.Tensor or PIL.Image.Image"
        )

    tensor = tensor.to(device)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]

    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k)

    results = []

    for val, idx in zip(values.tolist(), indices.tolist()):
        name = class_names[idx] if class_names else str(idx)
        results.append((name, val))

    return results


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")