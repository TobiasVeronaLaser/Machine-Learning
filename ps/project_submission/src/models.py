from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Baseline MLP.

    This model intentionally keeps the architecture simple:
    flattened grayscale image -> hidden layer -> output logits.

    It is used as a reference baseline without dropout or other regularization.
    """
    def __init__(self, input_size: int = 64 * 64, hidden_size: int = 512, num_classes: int = 36):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ImprovedMLP(nn.Module):
    """
    Improved MLP.

    Compared to the baseline, this model adds:
        - one additional hidden layer
        - dropout for regularization

    The model still uses flattened pixels, so it does not exploit spatial structure
    as directly as a CNN would.
    """
    def __init__(
        self,
        input_size: int = 64 * 64,
        hidden_size: int = 512,
        num_classes: int = 36,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
