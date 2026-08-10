"""Paired dense-to-sparse GPS knowledge distillation utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


def _sample_arrays(data, sample_id, noisy: bool):
    trajectory = data.trajectory_data
    feature = data.feature_data
    if noisy:
        x1 = trajectory.noise_feature_df.loc[sample_id].values.copy()
        x2 = feature.noise_feature_df.loc[sample_id].values.copy()
    else:
        x1 = trajectory.clean_feature_df.loc[sample_id].values.copy()
        x2 = feature.clean_feature_df.loc[sample_id].values.copy()
    return x1, x2


class CrossRatePairedDataset(Dataset):
    """Return aligned sparse student and dense teacher views of one window."""

    def __init__(self, student_data, teacher_data, indices, noise_probability=0.5):
        self.student_data = student_data
        self.teacher_data = teacher_data
        self.ids = list(indices)
        self.noise_probability = float(noise_probability)

        if student_data.pair_ids is None or teacher_data.pair_ids is None:
            raise ValueError(
                "Cross-rate distillation requires segment_pair_ids.npy and "
                "segment_time_ranges.npy")

        teacher_by_pair = {}
        for teacher_id in map(int, teacher_data.all_IDs):
            pair_id = str(teacher_data.pair_ids[teacher_id])
            teacher_by_pair.setdefault(pair_id, []).append(teacher_id)

        self.teacher_id_by_student_id = {}
        missing = []
        for student_id in map(int, self.ids):
            pair_id = str(student_data.pair_ids[student_id])
            candidates = teacher_by_pair.get(pair_id, [])
            if not candidates:
                missing.append(student_id)
                continue
            student_range = student_data.time_ranges[student_id]
            student_midpoint = float(student_range.mean())

            def pairing_score(teacher_id):
                teacher_range = teacher_data.time_ranges[teacher_id]
                overlap = max(
                    0.0,
                    min(student_range[1], teacher_range[1])
                    - max(student_range[0], teacher_range[0]),
                )
                midpoint_distance = abs(
                    student_midpoint - float(teacher_range.mean()))
                return overlap, -midpoint_distance

            self.teacher_id_by_student_id[student_id] = max(
                candidates, key=pairing_score)
        if missing:
            raise ValueError(
                f"Teacher data is missing {len(missing)} paired windows; "
                f"first missing ID: {missing[0]}"
            )

        student_labels = student_data.labels_df.loc[
            self.ids].values.reshape(-1)
        paired_teacher_ids = [
            self.teacher_id_by_student_id[int(item)] for item in self.ids]
        teacher_labels = teacher_data.labels_df.loc[
            paired_teacher_ids].values.reshape(-1)
        if not np.array_equal(student_labels, teacher_labels):
            raise ValueError("Student and teacher labels are not aligned")

    def __getitem__(self, index):
        sample_id = self.ids[index]
        student_noisy = random.random() < self.noise_probability
        student_x1, student_x2 = _sample_arrays(
            self.student_data, sample_id, student_noisy)
        # The dense teacher is a fixed target, so it always receives the clean
        # paired view. Student noise augmentation remains identical to B0.
        teacher_id = self.teacher_id_by_student_id[int(sample_id)]
        teacher_x1, teacher_x2 = _sample_arrays(
            self.teacher_data, teacher_id, noisy=False)
        label = self.student_data.labels_df.loc[sample_id].values.copy()
        return (
            torch.from_numpy(student_x1),
            torch.from_numpy(student_x2),
            torch.from_numpy(teacher_x1),
            torch.from_numpy(teacher_x2),
            torch.from_numpy(label),
            torch.as_tensor(sample_id),
        )

    def __len__(self):
        return len(self.ids)


def _pad_sequences(sequences):
    lengths = torch.as_tensor([item.shape[0] for item in sequences])
    max_len = int(lengths.max())
    output_dtype = (
        torch.float32
        if sequences[0].is_floating_point()
        else sequences[0].dtype
    )
    output = torch.zeros(
        len(sequences), max_len, sequences[0].shape[-1],
        dtype=output_dtype,
    )
    for index, sequence in enumerate(sequences):
        output[index, :sequence.shape[0], :] = sequence.to(output_dtype)
    positions = torch.arange(max_len).unsqueeze(0)
    padding_mask = positions < lengths.unsqueeze(1)
    return output, padding_mask


def collate_cross_rate_superv(batch):
    """Collate aligned student/teacher dual-branch samples."""
    student_x1, student_x2, teacher_x1, teacher_x2, labels, ids = zip(*batch)
    sx1, sm1 = _pad_sequences(student_x1)
    sx2, sm2 = _pad_sequences(student_x2)
    tx1, tm1 = _pad_sequences(teacher_x1)
    tx2, tm2 = _pad_sequences(teacher_x2)
    return [
        sx1, sx2, sm1, sm2,
        tx1, tx2, tm1, tm2,
        torch.stack(labels, dim=0), ids,
    ]


@dataclass
class DistillationTerms:
    total: Tensor
    supervised: Tensor
    logits: Tensor
    features: Tensor
    teacher_confidence: Tensor
    density_ratio: Tensor


def cross_rate_distillation_terms(
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Tensor,
        teacher_features: Tensor,
        targets: Tensor,
        loss_module,
        student_padding_mask: Tensor,
        teacher_padding_mask: Tensor,
        temperature: float = 4.0,
        logits_weight: float = 0.3,
        feature_weight: float = 0.05,
        confidence_power: float = 1.0,
        minimum_quality_weight: float = 0.25) -> DistillationTerms:
    """Compute supervised and sampling-quality-aware distillation losses.

    Teacher confidence suppresses uncertain soft targets. The observed point
    ratio measures how sparse the student view is relative to its paired dense
    view; more severely thinned views receive stronger teacher guidance.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= minimum_quality_weight <= 1:
        raise ValueError("minimum_quality_weight must be in [0, 1]")

    supervised = loss_module(student_logits, targets).mean()

    teacher_logits = teacher_logits.detach()
    teacher_features = teacher_features.detach()
    teacher_probability = torch.softmax(teacher_logits, dim=-1)
    teacher_confidence = teacher_probability.max(dim=-1).values

    student_count = student_padding_mask.sum(dim=1).to(student_logits.dtype)
    teacher_count = teacher_padding_mask.sum(dim=1).to(student_logits.dtype)
    density_ratio = (student_count / teacher_count.clamp_min(1.0)).clamp(0.0, 1.0)
    sparsity = 1.0 - density_ratio
    rate_weight = minimum_quality_weight + (
        1.0 - minimum_quality_weight) * sparsity
    quality_weight = teacher_confidence.pow(confidence_power) * rate_weight

    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    logits_per_sample = F.kl_div(
        log_student, soft_teacher, reduction="none").sum(dim=-1)
    logits = (quality_weight * logits_per_sample).mean() * temperature ** 2

    feature_per_sample = 1.0 - F.cosine_similarity(
        student_features, teacher_features, dim=-1)
    features = (quality_weight * feature_per_sample).mean()
    total = supervised + logits_weight * logits + feature_weight * features
    return DistillationTerms(
        total=total,
        supervised=supervised,
        logits=logits,
        features=features,
        teacher_confidence=teacher_confidence.mean(),
        density_ratio=density_ratio.mean(),
    )
