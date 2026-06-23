import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    """Linear multiclass baseline: flattened image -> 36 logits."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fc(x)


class Net(nn.Module):
    """MLP adapted from the course addendum.

    The network returns raw logits. Do not add softmax here when using
    nn.CrossEntropyLoss(), because CrossEntropyLoss applies LogSoftmax internally.
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
