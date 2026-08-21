"""Sparsity- and uncertainty-aware bounded Bus/Car correction."""

from __future__ import annotations

import torch
from torch import nn


def sparsity_gate(valid_points: torch.Tensor, n_low: float = 5.0,
                  n_high: float = 15.0) -> torch.Tensor:
    return ((n_high - valid_points) / (n_high - n_low)).clamp(0.0, 1.0)


def uncertainty_gate(bus_car_margin: torch.Tensor, tau: float = 1.0,
                     temperature: float = 0.5) -> torch.Tensor:
    return torch.sigmoid((tau - bus_car_margin.abs()) / temperature)


class ConservativeBusCarCorrector(nn.Module):
    """Predict one bounded antisymmetric update for Bus and Car logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 16,
                 alpha: float = 0.5, n_low: float = 5.0,
                 n_high: float = 15.0, uncertainty_tau: float = 1.0,
                 uncertainty_temperature: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.n_low = float(n_low)
        self.n_high = float(n_high)
        self.uncertainty_tau = float(uncertainty_tau)
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor,
                valid_points: torch.Tensor):
        margin = (base_logits[:, 2] - base_logits[:, 3]).abs()
        sparse = sparsity_gate(valid_points, self.n_low, self.n_high)
        uncertain = uncertainty_gate(
            margin, self.uncertainty_tau, self.uncertainty_temperature)
        delta = self.alpha * sparse * uncertain * torch.tanh(
            self.network(features).squeeze(-1))
        corrected = base_logits.clone()
        corrected[:, 2] = corrected[:, 2] + delta
        corrected[:, 3] = corrected[:, 3] - delta
        return corrected, delta, sparse, uncertain
