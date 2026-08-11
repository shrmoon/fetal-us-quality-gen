"""
Train a fetal ultrasound plane classifier.

Example:
    python src/train.py --data_dir data/ --epochs 15 --model resnet18
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FetalPlanesDataset
from model import get_model
from utils import set_seed, get_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "simple_cnn"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--ood_machines", type=str, default=None,
                    help="Comma-separated list of US_Machine values to hold out as OOD")
    p.add_argument("--output_dir", type=str, default="results/")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = os.path.join(args.data_dir, "FETAL_PLANES_DB_data.csv")
    images_dir = os.path.join(args.data_dir, "Images")

    ood_machines = args.ood_machines.split(",") if args.ood_machines else None

    train_set = FetalPlanesDataset(csv_path, images_dir, split="train",
                                    ood_machines=ood_machines,
                                    transform=get_transforms(train=True, image_size=args.image_size),
                                    seed=args.seed)
    id_test_set = FetalPlanesDataset(csv_path, images_dir, split="id_test",
                                      ood_machines=ood_machines,
                                      transform=get_transforms(train=False, image_size=args.image_size),
                                      seed=args.seed)

    print(f"Train size: {len(train_set)} | ID test size: {len(id_test_set)}")
    print(f"Held-out OOD machine(s): {train_set.ood_machines}")
    print(f"Classes: {train_set.classes}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    id_test_loader = DataLoader(id_test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(args.model, num_classes=len(train_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, id_test_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"ID-val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_set.classes,
                "ood_machines": train_set.ood_machines,
                "model_name": args.model,
                "image_size": args.image_size,
            }, checkpoint_path)
            print(f"  -> Saved new best model (ID-val acc {val_acc:.3f}) to {checkpoint_path}")

    print(f"\nTraining complete. Best ID-val accuracy: {best_acc:.3f}")
    print("Run src/evaluate.py to measure the ID vs OOD generalization gap.")


if __name__ == "__main__":
    main()
