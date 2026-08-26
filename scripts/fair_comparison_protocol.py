"""Single source of truth for the five-rate representative-method benchmark.

Every adapted model must load samples through this module and must return
probabilities in the original saved order.  User partitions, label ordering,
data fingerprints and metrics are therefore shared instead of reimplemented
inside each baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RATES = (5, 10, 20, 30, 60)
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("Walk", "Bike", "Bus", "Car", "Train")
MAX_POINTS = 64
# dt, distance, speed, acceleration, jerk, heading change, heading-change rate
MOTION_INDICES = (0, 2, 3, 4, 5, 7, 8)
DABIRI_INDICES = (3, 4, 5, 8)


@dataclass(frozen=True)
class SplitData:
    values: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    fingerprint: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _finite(array) -> np.ndarray:
    return np.nan_to_num(np.asarray(array, dtype=np.float64).reshape(-1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def _anchor_indices(length: int) -> np.ndarray:
    if length <= MAX_POINTS:
        return np.arange(length, dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, MAX_POINTS)).astype(np.int64)


def tensorize(feature_samples, trajectory_samples) -> tuple[np.ndarray, np.ndarray]:
    """Convert the saved object arrays to one GPS-only motion tensor.

    Channels are seven motion sequences plus metric relative x/y displacement.
    No GIS, POI, user identity or test-set statistic is introduced.
    """
    values = np.zeros((len(feature_samples), 9, MAX_POINTS), dtype=np.float32)
    mask = np.zeros((len(feature_samples), MAX_POINTS), dtype=bool)
    for row in range(len(feature_samples)):
        available = [len(_finite(feature_samples[row, i])) for i in MOTION_INDICES]
        available.extend((len(_finite(trajectory_samples[row, 0])),
                          len(_finite(trajectory_samples[row, 1]))))
        length = min(available)
        if length <= 0:
            continue
        indices = _anchor_indices(length)
        used = len(indices)
        channels = [_finite(feature_samples[row, i])[:length][indices]
                    for i in MOTION_INDICES]
        latitude = _finite(trajectory_samples[row, 0])[:length][indices]
        longitude = _finite(trajectory_samples[row, 1])[:length][indices]
        mean_latitude = math.radians(float(latitude.mean()))
        relative_y = (latitude - latitude[0]) * 111_320.0
        relative_x = ((longitude - longitude[0]) * 111_320.0
                      * math.cos(mean_latitude))
        channels.extend((relative_x, relative_y))
        values[row, :, :used] = np.stack(channels).astype(np.float32)
        mask[row, :used] = True
    return values, mask


def _sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_dir(rate: int, split: str) -> Path:
    if rate not in RATES or split not in SPLITS:
        raise ValueError(f"unsupported protocol coordinate: {rate}s/{split}")
    return ROOT / f"data/geolife_five_rate_fixed_{rate}s_features" / split


def load_split(rate: int, split: str, rebuild_cache: bool = False) -> SplitData:
    source = _source_dir(rate, split)
    cache = ROOT / "data/fair_comparison_cache" / f"{rate}s" / split
    cache.mkdir(parents=True, exist_ok=True)
    files = {
        "values": cache / "values.npy",
        "mask": cache / "mask.npy",
        "labels": cache / "labels.npy",
        "users": cache / "users.npy",
    }
    if rebuild_cache or not all(path.exists() for path in files.values()):
        features = np.load(source / "clean_multi_feature_segs.npy", allow_pickle=True)
        trajectories = np.load(source / "clean_trj_segs.npy", allow_pickle=True)
        labels = np.load(source / "clean_multi_feature_seg_labels.npy").astype(np.int64)
        noise_labels = np.load(source / "noise_multi_feature_seg_labels.npy").astype(np.int64)
        users = np.load(source / "segment_user_ids.npy").astype(np.int64)
        if not np.array_equal(labels, noise_labels):
            raise ValueError(f"clean/noisy labels differ at {rate}s/{split}")
        values, mask = tensorize(features, trajectories)
        if not (len(values) == len(mask) == len(labels) == len(users)):
            raise ValueError(f"sample count mismatch at {rate}s/{split}")
        np.save(files["values"], values)
        np.save(files["mask"], mask)
        np.save(files["labels"], labels)
        np.save(files["users"], users)
    values = np.load(files["values"], mmap_mode="r")
    mask = np.load(files["mask"], mmap_mode="r")
    labels = np.load(files["labels"])
    users = np.load(files["users"])
    fingerprint = _sha256(labels, users)
    return SplitData(values, mask, labels, users, fingerprint)


def validate_protocol(rebuild_cache: bool = False) -> dict:
    manifest = {"rates": {}, "class_names": list(CLASS_NAMES)}
    reference_users = None
    for rate in RATES:
        current = {split: load_split(rate, split, rebuild_cache) for split in SPLITS}
        user_sets = {split: set(data.users.tolist()) for split, data in current.items()}
        if any(user_sets[a] & user_sets[b]
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"user leakage detected at {rate}s")
        serialized_users = {split: sorted(users) for split, users in user_sets.items()}
        if reference_users is None:
            reference_users = serialized_users
        elif serialized_users != reference_users:
            raise ValueError(f"user partitions differ at {rate}s")
        manifest["rates"][str(rate)] = {
            split: {
                "samples": int(len(data.labels)),
                "users": len(user_sets[split]),
                "label_user_sha256": data.fingerprint,
                "class_counts": np.bincount(data.labels, minlength=5).tolist(),
            } for split, data in current.items()
        }
    manifest["user_partitions"] = reference_users
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def fit_channel_normalizer(data: SplitData) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(data.mask)
    values = np.asarray(data.values)
    means, stds = [], []
    for channel in range(values.shape[1]):
        selected = values[:, channel][valid]
        means.append(float(selected.mean()))
        stds.append(max(float(selected.std()), 1e-6))
    return (np.asarray(means, dtype=np.float32),
            np.asarray(stds, dtype=np.float32))


def normalize_values(data: SplitData, mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    values = (np.asarray(data.values, dtype=np.float32)
              - mean[None, :, None]) / std[None, :, None]
    return values * np.asarray(data.mask)[:, None, :]


def summarize(data: SplitData) -> np.ndarray:
    """Training-independent statistics for RF/XGBoost/DeepInsight."""
    values, masks = np.asarray(data.values), np.asarray(data.mask)
    output = []
    for row in range(len(values)):
        length = int(masks[row].sum())
        stats = []
        for channel in values[row, :, :length]:
            stats.extend((np.mean(channel), np.std(channel), np.min(channel),
                          np.max(channel), np.median(channel),
                          np.quantile(channel, .25), np.quantile(channel, .75)))
        displacement = math.hypot(float(values[row, -2, max(length - 1, 0)]),
                                  float(values[row, -1, max(length - 1, 0)]))
        stats.extend((length, displacement))
        output.append(stats)
    return np.nan_to_num(np.asarray(output, dtype=np.float32))


def evaluate_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """The only permitted Accuracy/Macro-F1 implementation in this benchmark."""
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), len(CLASS_NAMES)):
        raise ValueError(f"invalid probability shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite prediction probabilities")
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "per_class_f1": {
            name: float(score) for name, score in zip(
                CLASS_NAMES,
                f1_score(labels, prediction, labels=np.arange(5), average=None,
                         zero_division=0),
            )
        },
    }
