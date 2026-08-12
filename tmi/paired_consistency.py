"""Paired multi-rate training utilities for V6."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


def _sample_arrays(data, sample_id, noisy):
    trajectory = data.trajectory_data
    feature = data.feature_data
    if noisy:
        x1 = trajectory.noise_feature_df.loc[sample_id].values.copy()
        x2 = feature.noise_feature_df.loc[sample_id].values.copy()
    else:
        x1 = trajectory.clean_feature_df.loc[sample_id].values.copy()
        x2 = feature.clean_feature_df.loc[sample_id].values.copy()
    return x1, x2


class PairedMultiRateDataset(Dataset):
    """Return two aligned rates of one physical window for one shared model."""

    def __init__(self, sparse_data, dense_data, indices, noise_probability=0.5):
        self.sparse_data = sparse_data
        self.dense_data = dense_data
        self.ids = list(indices)
        self.noise_probability = float(noise_probability)
        if sparse_data.pair_ids is None or dense_data.pair_ids is None:
            raise ValueError(
                "Paired consistency requires segment_pair_ids.npy and "
                "segment_time_ranges.npy")

        dense_by_pair = {}
        for dense_id in map(int, dense_data.all_IDs):
            dense_by_pair.setdefault(str(dense_data.pair_ids[dense_id]), []).append(dense_id)

        self.dense_id_by_sparse_id = {}
        missing = []
        for sparse_id in map(int, self.ids):
            candidates = dense_by_pair.get(str(sparse_data.pair_ids[sparse_id]), [])
            if not candidates:
                missing.append(sparse_id)
                continue
            sparse_range = sparse_data.time_ranges[sparse_id]
            sparse_midpoint = float(sparse_range.mean())

            def score(dense_id):
                dense_range = dense_data.time_ranges[dense_id]
                overlap = max(
                    0.0,
                    min(sparse_range[1], dense_range[1])
                    - max(sparse_range[0], dense_range[0]),
                )
                midpoint_distance = abs(sparse_midpoint - float(dense_range.mean()))
                return overlap, -midpoint_distance

            self.dense_id_by_sparse_id[sparse_id] = max(candidates, key=score)
        if missing:
            raise ValueError(
                f"Dense view is missing {len(missing)} paired windows; "
                f"first missing ID: {missing[0]}")

        sparse_labels = sparse_data.labels_df.loc[self.ids].values.reshape(-1)
        dense_ids = [self.dense_id_by_sparse_id[int(item)] for item in self.ids]
        dense_labels = dense_data.labels_df.loc[dense_ids].values.reshape(-1)
        if not np.array_equal(sparse_labels, dense_labels):
            raise ValueError("Sparse and dense labels are not aligned")

    def __getitem__(self, index):
        sparse_id = self.ids[index]
        dense_id = self.dense_id_by_sparse_id[int(sparse_id)]
        # One shared noise draw avoids making view consistency a noise-removal task.
        noisy = random.random() < self.noise_probability
        sparse_x1, sparse_x2 = _sample_arrays(self.sparse_data, sparse_id, noisy)
        dense_x1, dense_x2 = _sample_arrays(self.dense_data, dense_id, noisy)
        label = self.sparse_data.labels_df.loc[sparse_id].values.copy()
        return (
            torch.from_numpy(sparse_x1), torch.from_numpy(sparse_x2),
            torch.from_numpy(dense_x1), torch.from_numpy(dense_x2),
            torch.from_numpy(label), torch.as_tensor(sparse_id),
        )

    def __len__(self):
        return len(self.ids)


def _pad_sequences(sequences):
    lengths = torch.as_tensor([item.shape[0] for item in sequences])
    max_len = int(lengths.max())
    output = torch.zeros(
        len(sequences), max_len, sequences[0].shape[-1], dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        output[index, :sequence.shape[0]] = sequence.to(torch.float32)
    positions = torch.arange(max_len).unsqueeze(0)
    return output, positions < lengths.unsqueeze(1)


def collate_paired_multirate(batch):
    sparse_x1, sparse_x2, dense_x1, dense_x2, labels, ids = zip(*batch)
    sx1, sm1 = _pad_sequences(sparse_x1)
    sx2, sm2 = _pad_sequences(sparse_x2)
    dx1, dm1 = _pad_sequences(dense_x1)
    dx2, dm2 = _pad_sequences(dense_x2)
    return [
        sx1, sx2, sm1, sm2, dx1, dx2, dm1, dm2,
        torch.stack(labels, dim=0), ids,
    ]


@dataclass
class PairedConsistencyTerms:
    total: Tensor
    sparse_supervised: Tensor
    dense_supervised: Tensor
    consistency: Tensor
    ramp: float


def paired_consistency_terms(
        sparse_logits, dense_logits, targets, loss_module, epoch,
        sparse_supervised_weight=0.7, consistency_weight=0.2,
        ramp_epochs=10):
    """Combine two supervised views with symmetric Jensen-Shannon consistency."""
    if not 0.0 <= sparse_supervised_weight <= 1.0:
        raise ValueError("sparse_supervised_weight must be in [0, 1]")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    if ramp_epochs < 0:
        raise ValueError("ramp_epochs must be non-negative")

    sparse_supervised = loss_module(sparse_logits, targets).mean()
    dense_supervised = loss_module(dense_logits, targets).mean()
    sparse_probability = torch.softmax(sparse_logits, dim=-1)
    dense_probability = torch.softmax(dense_logits, dim=-1)
    mixture = 0.5 * (sparse_probability + dense_probability)
    consistency = 0.5 * (
        F.kl_div(torch.log(mixture.clamp_min(1e-8)), sparse_probability,
                 reduction="batchmean")
        + F.kl_div(torch.log(mixture.clamp_min(1e-8)), dense_probability,
                   reduction="batchmean")
    )
    ramp = 1.0 if ramp_epochs == 0 else min(1.0, max(0.0, float(epoch) / ramp_epochs))
    supervised = (
        sparse_supervised_weight * sparse_supervised
        + (1.0 - sparse_supervised_weight) * dense_supervised
    )
    total = supervised + consistency_weight * ramp * consistency
    return PairedConsistencyTerms(
        total=total,
        sparse_supervised=sparse_supervised,
        dense_supervised=dense_supervised,
        consistency=consistency,
        ramp=ramp,
    )
