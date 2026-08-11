import random
import numpy as np
import torch
from torchvision import transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train=True, image_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


def plot_generalization_gap(id_acc, ood_acc, save_path="results/generalization_gap.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["In-Distribution", "Out-of-Distribution"], [id_acc, ood_acc],
                   color=["#2c7fb8", "#d95f0e"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Generalization Gap: {(id_acc - ood_acc) * 100:.1f} pts")
    for bar, acc in zip(bars, [id_acc, ood_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                 f"{acc*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
