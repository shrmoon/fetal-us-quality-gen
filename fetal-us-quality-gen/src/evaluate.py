"""
Evaluate a trained model on both the in-distribution (ID) and
out-of-distribution (OOD) test sets, and report the generalization gap.

Example:
    python src/evaluate.py --data_dir data/ --checkpoint results/best_model.pt
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import FetalPlanesDataset
from model import get_model
from utils import get_transforms, plot_generalization_gap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--checkpoint", type=str, default="results/best_model.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", type=str, default="results/")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    ood_machines = checkpoint["ood_machines"]
    image_size = checkpoint["image_size"]

    csv_path = os.path.join(args.data_dir, "FETAL_PLANES_DB_data.csv")
    images_dir = os.path.join(args.data_dir, "Images")

    id_test_set = FetalPlanesDataset(csv_path, images_dir, split="id_test",
                                      ood_machines=ood_machines,
                                      transform=get_transforms(train=False, image_size=image_size))
    ood_test_set = FetalPlanesDataset(csv_path, images_dir, split="ood_test",
                                       ood_machines=ood_machines,
                                       transform=get_transforms(train=False, image_size=image_size))

    id_loader = DataLoader(id_test_set, batch_size=args.batch_size, shuffle=False)
    ood_loader = DataLoader(ood_test_set, batch_size=args.batch_size, shuffle=False)

    model = get_model(checkpoint["model_name"], num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Evaluating on ID test set ({len(id_test_set)} images)...")
    id_acc, id_report, id_cm = evaluate(model, id_loader, device, classes)

    print(f"Evaluating on OOD test set ({len(ood_test_set)} images, machine(s): {ood_machines})...")
    ood_acc, ood_report, ood_cm = evaluate(model, ood_loader, device, classes)

    print("\n" + "=" * 60)
    print(f"In-distribution accuracy:     {id_acc:.3f}")
    print(f"Out-of-distribution accuracy: {ood_acc:.3f}")
    print(f"Generalization gap:           {(id_acc - ood_acc):.3f}")
    print("=" * 60)

    print("\nID classification report:\n", id_report)
    print("\nOOD classification report:\n", ood_report)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_generalization_gap(id_acc, ood_acc,
                             save_path=os.path.join(args.output_dir, "generalization_gap.png"))


if __name__ == "__main__":
    main()
