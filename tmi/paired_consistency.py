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


def effective_number_class_weights(labels, num_classes, beta=0.9999):
    """Return mean-one class weights computed only from training labels."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if len(counts) != num_classes or np.any(counts <= 0):
        raise ValueError("Every class must occur in the paired training split")
    if beta == 0.0:
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
        weights /= weights.mean()
    return weights.astype(np.float32)


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
        ramp_epochs=10, class_weights=None,
        confidence_threshold=0.0, consistency_temperature=1.0):
    """Combine class-balanced supervision with confidence-gated consistency."""
    if not 0.0 <= sparse_supervised_weight <= 1.0:
        raise ValueError("sparse_supervised_weight must be in [0, 1]")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    if ramp_epochs < 0:
        raise ValueError("ramp_epochs must be non-negative")
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in [0, 1)")
    if consistency_temperature <= 0.0:
        raise ValueError("consistency_temperature must be positive")

    sparse_losses = loss_module(sparse_logits, targets)
    dense_losses = loss_module(dense_logits, targets)
    if class_weights is None:
        sample_weights = torch.ones_like(sparse_losses)
    else:
        class_weights = torch.as_tensor(
            class_weights, dtype=sparse_losses.dtype,
            device=sparse_losses.device,
        )
        if class_weights.ndim != 1 or len(class_weights) != sparse_logits.shape[-1]:
            raise ValueError("class_weights must contain one value per class")
        if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
            raise ValueError("class_weights must be finite and positive")
        sample_weights = class_weights[targets.reshape(-1).long()]
    normalizer = sample_weights.sum().clamp_min(1e-8)
    sparse_supervised = (sparse_losses * sample_weights).sum() / normalizer
    dense_supervised = (dense_losses * sample_weights).sum() / normalizer

    temperature = float(consistency_temperature)
    sparse_probability = torch.softmax(sparse_logits / temperature, dim=-1)
    dense_probability = torch.softmax(dense_logits / temperature, dim=-1)
    mixture = 0.5 * (sparse_probability + dense_probability)
    log_mixture = torch.log(mixture.clamp_min(1e-8))
    per_sample_consistency = 0.5 * (
        F.kl_div(log_mixture, sparse_probability, reduction="none").sum(dim=-1)
        + F.kl_div(log_mixture, dense_probability, reduction="none").sum(dim=-1)
    ) * (temperature ** 2)
    dense_confidence = dense_probability.detach().amax(dim=-1)
    confidence_gate = (
        (dense_confidence - confidence_threshold)
        / (1.0 - confidence_threshold)
    ).clamp(0.0, 1.0)
    gate_total = confidence_gate.sum()
    if gate_total.item() > 0.0:
        consistency = (
            per_sample_consistency * confidence_gate
        ).sum() / gate_total
    else:
        consistency = per_sample_consistency.sum() * 0.0
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


def representation_recovery_loss(
        sparse_representation, dense_representation, dense_logits,
        confidence_threshold=0.5, temperature=1.0):
    """Align recovered sparse features to a detached, confident dense view."""
    if sparse_representation.shape != dense_representation.shape:
        raise ValueError("Sparse and dense representations must have equal shape")
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in [0, 1)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    dense_target = dense_representation.detach()
    sparse_normalized = F.normalize(sparse_representation, dim=-1)
    dense_normalized = F.normalize(dense_target, dim=-1)
    per_sample = 1.0 - (
        sparse_normalized * dense_normalized
    ).sum(dim=-1)

    dense_probability = torch.softmax(
        dense_logits.detach() / float(temperature), dim=-1
    )
    confidence = dense_probability.amax(dim=-1)
    gate = (
        (confidence - confidence_threshold)
        / (1.0 - confidence_threshold)
    ).clamp(0.0, 1.0)
    gate_sum = gate.sum()
    if gate_sum.item() > 0.0:
        loss = (per_sample * gate).sum() / gate_sum
    else:
        loss = per_sample.sum() * 0.0
    coverage = (gate > 0.0).to(per_sample.dtype).mean()
    return loss, coverage
