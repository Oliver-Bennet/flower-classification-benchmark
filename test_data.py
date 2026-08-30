from datasets.flower_dataset import FlowerDataset, get_transforms
from torch.utils.data import DataLoader
import yaml

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    train_tf, val_tf = get_transforms()

    train_dataset = FlowerDataset(
        config["dataset"]["root"],
        transform=train_tf,
        split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=4
    )

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)

if __name__ == "__main__":
    main()