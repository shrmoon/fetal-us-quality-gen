"""
Dataset loading and ID/OOD (in-distribution / out-of-distribution) splitting
for the FETAL_PLANES_DB dataset.

The core idea of this project: instead of a random train/test split, we
deliberately hold out one or more `US_Machine` values entirely from
training, so the OOD test set represents a genuine domain shift (a machine
the model has never seen), while the ID test set is a random held-out split
from machines the model *has* seen during training.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# If your downloaded CSV uses different column names, update these:
COL_IMAGE = "Image_name"
COL_PLANE = "Plane"
COL_MACHINE = "US_Machine"
COL_OPERATOR = "Operator"
COL_TRAIN = "Train"


class FetalPlanesDataset(Dataset):
    def __init__(self, csv_path, images_dir, split="train", ood_machines=None,
                 transform=None, seed=42):
        """
        Args:
            csv_path: path to FETAL_PLANES_DB_data.csv
            images_dir: path to the folder containing the .png images
            split: one of "train", "id_test", "ood_test"
            ood_machines: list of US_Machine values to hold out entirely as
                OOD. If None, defaults to holding out the least common
                machine in the dataset.
            transform: torchvision transform to apply to images
            seed: random seed for the ID train/test split
        """
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(csv_path, sep=";")

        # Build label mapping from the Plane column
        self.classes = sorted(df[COL_PLANE].dropna().unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if ood_machines is None:
            # Default: hold out the least-represented machine as OOD
            machine_counts = df[COL_MACHINE].value_counts()
            ood_machines = [machine_counts.idxmin()]
        self.ood_machines = ood_machines

        is_ood_machine = df[COL_MACHINE].isin(ood_machines)

        if split == "ood_test":
            self.df = df[is_ood_machine].reset_index(drop=True)
        else:
            # Everything from non-OOD machines gets split into train / id_test
            in_domain_df = df[~is_ood_machine].reset_index(drop=True)
            shuffled = in_domain_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            split_point = int(0.85 * len(shuffled))
            if split == "train":
                self.df = shuffled.iloc[:split_point].reset_index(drop=True)
            elif split == "id_test":
                self.df = shuffled.iloc[split_point:].reset_index(drop=True)
            else:
                raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[COL_IMAGE]
        # Images are typically stored as .png; adjust if your download differs
        img_path = os.path.join(self.images_dir, f"{img_name}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row[COL_PLANE]]
        return image, label
