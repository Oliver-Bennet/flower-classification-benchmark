
import pandas as pd
import torch
from PIL import Image


class Predictor:

    def __init__(
        self,
        model,
        checkpoint,
        transform,
        csv_path,
        device
    ):

        self.device = device

        self.model = model.to(device)

        ckpt = torch.load(
            checkpoint,
            map_location=device
        )

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

        self.transform = transform

        self.df = pd.read_csv(csv_path)

    @torch.no_grad()
    def predict(self, image):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(x)

        probs = torch.softmax(logits, dim=1)

        conf, pred = torch.max(probs, dim=1)

        idx = pred.item()

        flower_id = flower_id = idx + 1

        info = self.df[self.df.id == flower_id].iloc[0]

        return {
            "id": flower_id,
            "name": info["name"],
            "confidence": conf.item()
        }

    @torch.no_grad()
    def predict_topk(self, image, k=5):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)

        probs = torch.softmax(
            self.model(x),
            dim=1
        )

        values, indices = torch.topk(
            probs,
            k=k
        )

        result = []

        for p, idx in zip(values[0], indices[0]):

            flower_id = flower_id = idx.item() + 1

            info = self.df[
                self.df.id == flower_id
            ].iloc[0]

            result.append({
                "id": flower_id,
                "name": info["name"],
                "confidence": float(p)
            })

        return result

    def get_flower_info(self, flower_id):

        row = self.df[
            self.df.id == flower_id
        ].iloc[0]

        return row.to_dict()