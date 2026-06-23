from __future__ import annotations

import torch
import torch.nn as nn  # neural network module from pytroch


class SimpleMLP(nn.Module):
    """
    Baseline MLP.
    flattened grayscale image -> hidden layer -> output logits.
    It is used as a reference baseline without optimization.
    """
    def __init__(self, input_size: int = 64 * 64, hidden_size: int = 512, num_classes: int = 36):
        super().__init__() 
        self.network = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(input_size, hidden_size),  
            nn.ReLU(),  # negativ values are set to 0
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
    The model still uses flattened pixels (CNN uses spatial information, which is not captured here).
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
            nn.ReLU(),  # negative values are set to 0
            nn.Dropout(dropout), # prevents overfitting by randomly setting a fraction of input units to 0 during training
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)