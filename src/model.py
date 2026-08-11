"""
Model definitions: a fine-tuned ResNet18 (recommended) and a small CNN
baseline for comparison.
"""

import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classification layer for our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class SimpleCNN(nn.Module):
    """A small from-scratch CNN baseline, useful for comparing against the
    pretrained ResNet18 to see how much transfer learning helps."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(name, num_classes, pretrained=True):
    if name == "resnet18":
        return get_resnet18(num_classes, pretrained=pretrained)
    elif name == "simple_cnn":
        return SimpleCNN(num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
