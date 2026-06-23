from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


def get_device() -> torch.device:
    """
    Select the best available device.
    On most Windows laptops without NVIDIA CUDA this will return CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    idx_to_label: Optional[dict[int, str]] = None,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_targets.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    result = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    if idx_to_label is not None:
        labels_sorted = list(range(len(idx_to_label)))
        target_names = [idx_to_label[i] for i in labels_sorted]
        result["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
        )

    return result


def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    num_epochs: int,
) -> Dict[str, list]:
    """
    Simple fixed-epoch training used for the baseline experiments.
    """
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])
        history["val_accuracy"].append(val_result["accuracy"])
        history["val_macro_f1"].append(val_result["macro_f1"])

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_acc={val_result['accuracy']:.4f} | "
            f"val_macro_f1={val_result['macro_f1']:.4f}"
        )

    return history


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    save_path: str | Path,
) -> Dict[str, list]:
    """
    Improved training loop.

    The model with the lowest validation loss is saved to save_path.
    Training stops if validation loss does not improve for 'patience' epochs.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])
        history["val_accuracy"].append(val_result["accuracy"])
        history["val_macro_f1"].append(val_result["macro_f1"])

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_acc={val_result['accuracy']:.4f} | "
            f"val_macro_f1={val_result['macro_f1']:.4f}"
        )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  saved new best model to {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    return history


def plot_training_curves(history: Dict[str, list], output_path: str | Path, title: str = "Training curves"):
    """
    Save loss and validation metric curves.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metric_curve(history: Dict[str, list], output_path: str | Path, metric: str = "val_macro_f1"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history[metric]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[metric], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path: str | Path,
    title: str = "Confusion matrix",
):
    """
    Save a confusion matrix plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
