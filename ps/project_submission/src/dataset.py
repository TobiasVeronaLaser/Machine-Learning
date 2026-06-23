from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def load_labels(data_root: str | Path) -> pd.DataFrame:
    """Load labels.csv from the dataset folder.
    """
    data_root = Path(data_root)
    labels_path = data_root / "labels.csv"

    df = pd.read_csv(labels_path, header=None)

    df = df.iloc[:, :2].copy()
    df.columns = ["label", "filename"]

    df["label"] = df["label"].astype(str)
    df["filename"] = df["filename"].astype(str)

    def resolve_image_path(filename: str) -> str:
        filename_path = Path(filename)

        candidates = [
            data_root / filename_path,
            data_root / "images" / filename_path,
            data_root / "imgs" / filename_path,
            data_root / "train" / filename_path,
            data_root / "sign_lang_train" / filename_path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            f"Could not find image file for filename '{filename}'. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    df["image_path"] = df["filename"].apply(resolve_image_path)

    return df[["image_path", "label"]]


def make_label_mapping(labels) -> Tuple[dict[str, int], dict[int, str]]:
    """Create label mappings.
    """
    classes = sorted(pd.Series(labels).astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def make_transforms(image_size: int = 64, augment: bool = False):
    """    Preprocessing pipeline.

    - grayscale
    - resize
    - tensor
    - normalize
    - small random rotations/translations
    """
    tfms = [transforms.Grayscale(num_output_channels=1), transforms.Resize((image_size, image_size))]

    if augment:
        tfms.extend([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])

    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return transforms.Compose(tfms)


class SignLanguageDataset(Dataset):
    """
    PyTorch dataset for the sign-language image classification task.
    Returns:
        image tensor with shape [1, image_size, image_size]
        label index in [0, num_classes-1]
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_idx: Optional[dict[str, int]] = None,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        if label_to_idx is None:
            label_to_idx, _ = make_label_mapping(self.df["label"])
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_idx[str(row["label"])]
        return image, torch.tensor(label, dtype=torch.long)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified train/validation/test split.

    test_size: fraction of full data used for final test set.
    val_size: fraction of remaining train_val data used for validation.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights(train_df: pd.DataFrame, label_to_idx: dict[str, int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Rare classes receive larger weights. The weights are normalized to mean 1.
    """
    counts = train_df["label"].astype(str).value_counts()
    weights = np.zeros(len(label_to_idx), dtype=np.float32)

    for label, idx in label_to_idx.items():
        count = counts.get(label, 0)
        if count <= 0:
            weights[idx] = 0.0
        else:
            weights[idx] = 1.0 / float(count)

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
