from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDatasetFromPolars(Dataset):
    def __init__(self, df, image_root, transform=None, label_to_idx=None):
        self.df = df
        self.image_root = Path(image_root)
        self.transform = transform

        if label_to_idx is None:
            labels = self.df["label"].unique().sort().to_list()
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        image_path = self.image_root / row["image_path"]

        # Schwarz-Weiß-Bild: 1 Kanal
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_idx[row["label"]]

        return image, torch.tensor(label, dtype=torch.long)