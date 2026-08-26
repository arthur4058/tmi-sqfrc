"""Controlled reimplementations/adapters for representative GPS baselines."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DabiriCNN(nn.Module):
    """1-D CNN over speed, acceleration, jerk and bearing-change rate."""
    def __init__(self, classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.35),
                                        nn.Linear(128, classes))

    def forward(self, values, mask=None):
        return self.classifier(self.features(values[:, (2, 3, 4, 6)]))


class ChannelAttention1d(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(4, channels // ratio)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                 nn.Conv1d(channels, hidden, 1), nn.LeakyReLU(),
                                 nn.Conv1d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, values):
        return values * self.net(values)


class ScaleBranch(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 64, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.LeakyReLU(), ChannelAttention1d(128),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 128),
            nn.LeakyReLU())

    def forward(self, values):
        return self.net(values)


class MASOMSFAdapter(nn.Module):
    """MSF channel/scale-fusion architecture on the common motion tensor.

    The authors' user-random MASO image generation is deliberately not reused;
    doing so would violate the common window protocol.  Three temporal scales
    replace the three MASO scales while retaining channel and scale attention.
    """
    def __init__(self, classes: int = 5):
        super().__init__()
        self.branches = nn.ModuleList((ScaleBranch(3), ScaleBranch(5),
                                       ScaleBranch(9)))
        self.scale_attention = nn.Sequential(nn.Linear(384, 384), nn.Tanh(),
                                             nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.Linear(384, 128), nn.LeakyReLU(),
                                        nn.Dropout(.5), nn.Linear(128, classes))

    def forward(self, values, mask=None):
        fused = torch.cat([branch(values) for branch in self.branches], dim=1)
        return self.classifier(fused * self.scale_attention(fused))


class PatchEmbedding(nn.Module):
    def __init__(self, size: int, patch: int, dim: int):
        super().__init__()
        self.projection = nn.Conv2d(1, dim, patch, stride=patch)
        self.tokens = (size // patch) ** 2

    def forward(self, image):
        return self.projection(image).flatten(2).transpose(1, 2)


class DeepInsightViT(nn.Module):
    """ViT classifier for train-only DeepInsight-style feature images."""
    def __init__(self, image_size: int, classes: int = 5, dim: int = 96):
        super().__init__()
        patch = 2 if image_size % 2 == 0 else 1
        self.patch = PatchEmbedding(image_size, patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, self.patch.tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 3, dropout=.15, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3, nn.LayerNorm(dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

    def forward(self, image, mask=None):
        tokens = self.patch(image)
        cls = self.cls.expand(len(image), -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), 1) + self.position)
        return self.head(encoded[:, 0])


class FeatureImageMapper:
    """Deterministic, training-only correlation embedding of tabular features."""
    def __init__(self):
        self.mean = None
        self.std = None
        self.rows = None
        self.cols = None
        self.size = None

    def fit(self, features: np.ndarray):
        self.mean = features.mean(0)
        self.std = np.maximum(features.std(0), 1e-6)
        normalized = (features - self.mean) / self.std
        correlation = np.nan_to_num(np.corrcoef(normalized, rowvar=False))
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        coordinates = eigenvectors[:, -2:] * np.sqrt(
            np.maximum(eigenvalues[-2:], 0))[None]
        count = features.shape[1]
        self.size = int(math.ceil(math.sqrt(count)))
        ranks = np.empty_like(coordinates, dtype=np.int64)
        for axis in range(2):
            ranks[np.argsort(coordinates[:, axis]), axis] = np.arange(count)
        proposed = np.rint(ranks * (self.size - 1) / max(1, count - 1)).astype(int)
        free = {(row, col) for row in range(self.size) for col in range(self.size)}
        assigned = []
        for row, col in proposed:
            choice = min(free, key=lambda cell: (cell[0] - row) ** 2
                         + (cell[1] - col) ** 2)
            assigned.append(choice)
            free.remove(choice)
        self.rows = np.asarray([item[0] for item in assigned])
        self.cols = np.asarray([item[1] for item in assigned])
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("feature image mapper is not fitted")
        normalized = (features - self.mean) / self.std
        images = np.zeros((len(features), 1, self.size, self.size), np.float32)
        images[:, 0, self.rows, self.cols] = normalized
        return images
