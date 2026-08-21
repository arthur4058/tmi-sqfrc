"""Lightweight dense-privileged interval behavior recovery."""

from __future__ import annotations

import torch
from torch import nn


class IntervalBehaviorRecovery(nn.Module):
    """Predict physical dense-interval summaries from sparse observations."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32,
                 output_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, sparse_interval_features: torch.Tensor) -> torch.Tensor:
        return self.network(sparse_interval_features)
