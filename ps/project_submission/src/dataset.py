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
    """Load labels.csv from the dataset folder."""
    data_root = Path(data_root) 
    labels_path = data_root / "labels.csv"

    df = pd.read_csv(labels_path, header=None)

    df = df.iloc[:, :2].copy()  #   Takes the two columns 
    df.columns = ["label", "filename"]

    df["label"] = df["label"].astype(str)
    df["filename"] = df["filename"].astype(str)

    df["image_path"] = df["filename"].apply(lambda filename: str(data_root / filename))

    return df[["image_path", "label"]]


def make_label_mapping(labels) -> Tuple[dict[str, int], dict[int, str]]:
    """Create label mappings."""
    classes = sorted(pd.Series(labels).astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}    #   gives a label to id mapping
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}  #   gives a id to label mapping
    return label_to_idx, idx_to_label


def make_transforms(image_size: int = 64, augment: bool = False):
    """
    Preprocessing pipeline.

    - grayscale
    - resize
    - tensor
    - normalize
    - small random rotations/translations
    """
    tfms = [transforms.Grayscale(num_output_channels=1), transforms.Resize((image_size, image_size))]   # makes images grayscale and resizes them to the specified image_size

    if augment: #   If augment is True, add random data transformations.
        tfms.extend([
            transforms.RandomRotation(degrees=10),  # random rotation of the image by up to 10 degrees
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # random affine transformation with small translation and scaling
        ])

    tfms.extend([  # add final transformations to the pipeline
        transforms.ToTensor(),  # image to pytorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # normalize the tensor to have mean 0.5 and std 0.5
    ])

    return transforms.Compose(tfms)  # put tranformation into a pipeline


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
        if label_to_idx is None:  # create mapping if not provided
            label_to_idx, _ = make_label_mapping(self.df["label"])  
        self.label_to_idx = label_to_idx 

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int): 
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L") # open image and convert to grayscale

        if self.transform is not None: 
            image = self.transform(image)  # ammplies transformation 

        label = self.label_to_idx[str(row["label"])]
        return image, torch.tensor(label, dtype=torch.long) 


def stratified_split(   # split to train, validation and test sets while maintaining class distribution
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42, # seed for reproducibility
):
    """
    Stratified train/validation/test split.

    test_size: fraction of full data used for final test set.
    val_size: fraction of remaining train_val data used for validation.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],   # keep similar class distribution in the test set
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        stratify=train_val_df["label"],  # keep similar class distribution in the validation set
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