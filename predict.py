import argparse

import torch

from models.convnext import create_convnext
from datasets.flower_dataset import get_transforms
from utils.inference import Predictor


parser = argparse.ArgumentParser()

parser.add_argument("--image", required=True)

args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_, transform = get_transforms()

model = create_convnext(
    num_classes=102
)

predictor = Predictor(
    model=model,
    checkpoint="outputs/checkpoints/best_model.pth",
    transform=transform,
    csv_path="flower_info/flower_info.csv",
    device=device
)

result = predictor.predict(args.image)

print(result)