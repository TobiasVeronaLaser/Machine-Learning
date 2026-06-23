from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class SignLanguageDataset(Dataset):
    """PyTorch Dataset for the sign-language image dataset.

    Expected dataframe columns:
    - label: class label, e.g. "a", "7"
    - image_path: filename relative to image_root
    """

    def __init__(self, df, image_root, transform=None, label_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

        if label_to_idx is None:
            labels = sorted(self.df["label"].unique().tolist())
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image_path"]

        # Grayscale image: 1 channel
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_idx[row["label"]]
        return image, torch.tensor(label, dtype=torch.long)
