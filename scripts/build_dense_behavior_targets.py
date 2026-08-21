#!/usr/bin/env python3
"""Build leakage-free dense behavior targets for sparse GeoLife intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


TARGET_NAMES = (
    "mean_speed",
    "std_speed",
    "max_speed",
    "stop_ratio",
    "acceleration_energy",
    "turning_energy",
)
INPUT_NAMES = (
    "delta_t",
    "distance",
    "average_speed",
    "heading",
    "heading_change",
    "heading_change_rate",
    "valid_interval",
)
STOP_DISTANCE_METERS = 0.5
EARTH_RADIUS_METERS = 6_371_008.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=5)


def select_dev_users(labels: np.ndarray, user_ids: np.ndarray, seed: int,
                     dev_ratio: float = 0.2, trials: int = 20_000):
    """Choose a deterministic class-balanced Dev-A subset of training users."""
    users = np.unique(user_ids)
    n_dev = min(max(int(round(len(users) * dev_ratio)), 1), len(users) - 1)
    per_user = {
        int(user): class_counts(labels[user_ids == user]) for user in users
    }
    global_dist = class_counts(labels).astype(float)
    global_dist /= global_dist.sum()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        perm = rng.permutation(users)
        dev = np.sort(perm[:n_dev])
        train = np.sort(perm[n_dev:])
        dev_counts = np.sum([per_user[int(user)] for user in dev], axis=0)
        train_counts = np.sum([per_user[int(user)] for user in train], axis=0)
        if np.any(dev_counts == 0) or np.any(train_counts == 0):
            continue
        score = float(np.mean(np.abs(dev_counts / dev_counts.sum() - global_dist)))
        score += float(np.mean(np.abs(train_counts / train_counts.sum() - global_dist)))
        candidate = (score, train, dev, train_counts, dev_counts)
        if best is None or score < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not build a Train-A/Dev-A split covering all classes")
    return best[1], best[2]


def _edge_features(trajectory: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the repository-compatible physical edge quantities."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) < 2:
        empty = np.empty(0, dtype=np.float64)
        return {name: empty for name in (
            "start", "end", "dt", "distance", "speed", "acceleration",
            "heading", "heading_change")}

    start = trajectory[:-1, 0]
    end = trajectory[1:, 0]
    dt = end - start
    lat1 = np.radians(trajectory[:-1, 1])
    lat2 = np.radians(trajectory[1:, 1])
    dlat = lat2 - lat1
    dlon = np.radians(trajectory[1:, 2] - trajectory[:-1, 2])
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * EARTH_RADIUS_METERS * np.arcsin(np.minimum(1.0, np.sqrt(hav)))
    speed = np.divide(distance, dt, out=np.zeros_like(distance), where=dt > 0)

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    heading = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    previous_heading = np.concatenate(([0.0], heading[:-1]))
    heading_change = np.abs(heading - previous_heading)
    previous_speed = np.concatenate(([0.0], speed[:-1]))
    acceleration = np.divide(
        speed - previous_speed, dt, out=np.zeros_like(speed), where=dt > 0)
    return {
        "start": start,
        "end": end,
        "dt": dt,
        "distance": distance,
        "speed": speed,
        "acceleration": acceleration,
        "heading": heading,
        "heading_change": heading_change,
    }


def build_window_intervals(dense: np.ndarray, sparse: np.ndarray):
    dense_edges = _edge_features(dense)
    sparse_edges = _edge_features(sparse)
    n_intervals = len(sparse_edges["dt"])
    inputs = np.zeros((n_intervals, len(INPUT_NAMES)), dtype=np.float32)
    targets = np.full((n_intervals, len(TARGET_NAMES)), np.nan, dtype=np.float32)
    valid = np.zeros(n_intervals, dtype=bool)

    positive = sparse_edges["dt"] > 0
    inputs[:, 0] = sparse_edges["dt"]
    inputs[:, 1] = sparse_edges["distance"]
    inputs[:, 2] = sparse_edges["speed"]
    inputs[:, 3] = sparse_edges["heading"]
    inputs[:, 4] = sparse_edges["heading_change"]
    inputs[:, 5] = np.divide(
        sparse_edges["heading_change"], sparse_edges["dt"],
        out=np.zeros(n_intervals), where=positive)
    inputs[:, 6] = positive.astype(np.float32)

    for index, (left, right) in enumerate(zip(sparse_edges["start"], sparse_edges["end"])):
        edge_mask = (
            (dense_edges["start"] >= left)
            & (dense_edges["end"] <= right)
            & (dense_edges["dt"] > 0)
        )
        if np.count_nonzero(edge_mask) < 1:
            continue
        speeds = dense_edges["speed"][edge_mask]
        distances = dense_edges["distance"][edge_mask]
        accelerations = dense_edges["acceleration"][edge_mask]
        turns = dense_edges["heading_change"][edge_mask]
        targets[index] = (
            float(np.mean(speeds)),
            float(np.std(speeds)),
            float(np.max(speeds)),
            float(np.mean(distances < STOP_DISTANCE_METERS)),
            float(np.mean(np.abs(accelerations))),
            float(np.mean(np.abs(turns))),
        )
        valid[index] = True
    return inputs, targets, valid


def _load(root: Path, condition: str, name: str):
    return np.load(root / condition / f"train_{name}.npy", allow_pickle=True)


def _summary(values: np.ndarray) -> dict:
    return {
        name: {
            "mean": float(values[:, index].mean()),
            "std": float(values[:, index].std()),
            "min": float(values[:, index].min()),
            "max": float(values[:, index].max()),
        }
        for index, name in enumerate(TARGET_NAMES)
    }


def _build_split(name: str, indices: np.ndarray, dense_trjs, sparse_trjs,
                 labels, users, pairs, output_dir: Path):
    all_inputs = []
    all_targets = []
    all_valid = []
    all_windows = []
    all_intervals = []
    for local_window, source_index in enumerate(indices):
        inputs, targets, valid = build_window_intervals(
            dense_trjs[source_index], sparse_trjs[source_index])
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_valid.append(valid)
        all_windows.append(np.full(len(inputs), local_window, dtype=np.int32))
        all_intervals.append(np.arange(len(inputs), dtype=np.int16))

    inputs = np.concatenate(all_inputs) if all_inputs else np.empty((0, len(INPUT_NAMES)))
    targets = np.concatenate(all_targets) if all_targets else np.empty((0, len(TARGET_NAMES)))
    valid = np.concatenate(all_valid) if all_valid else np.empty(0, dtype=bool)
    window_index = np.concatenate(all_windows) if all_windows else np.empty(0, dtype=np.int32)
    interval_index = np.concatenate(all_intervals) if all_intervals else np.empty(0, dtype=np.int16)
    output_path = output_dir / f"{name}.npz"
    np.savez_compressed(
        output_path,
        sparse_features=inputs,
        behavior_targets=targets,
        valid_targets=valid,
        window_index=window_index,
        interval_index=interval_index,
        pair_ids=np.asarray(pairs[indices]),
        user_ids=np.asarray(users[indices]),
        labels=np.asarray(labels[indices]),
        source_indices=np.asarray(indices),
    )
    valid_values = targets[valid]
    return {
        "artifact": str(output_path),
        "artifact_sha256": sha256(output_path),
        "user_count": int(len(np.unique(users[indices]))),
        "users": list(map(int, np.unique(users[indices]))),
        "window_count": int(len(indices)),
        "class_counts": {
            str(label): int(count)
            for label, count in sorted(Counter(map(int, labels[indices])).items())
        },
        "sparse_interval_count": int(len(valid)),
        "valid_dense_target_intervals": int(valid.sum()),
        "invalid_dense_target_intervals": int((~valid).sum()),
        "valid_ratio": float(valid.mean()) if len(valid) else 0.0,
        "target_summary": _summary(valid_values),
    }, valid_values


def build(args):
    root = args.views_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_condition = "fixed_5s"
    sparse_condition = "fixed_60s"
    dense_trjs = _load(root, dense_condition, "trjs")
    sparse_trjs = _load(root, sparse_condition, "trjs")
    dense_meta = {
        name: _load(root, dense_condition, name)
        for name in ("labels", "user_ids", "source_ids", "pair_ids")
    }
    sparse_meta = {
        name: _load(root, sparse_condition, name)
        for name in ("labels", "user_ids", "source_ids", "pair_ids")
    }
    if len(dense_trjs) != len(sparse_trjs):
        raise AssertionError("5s/60s training window counts differ")
    for name in dense_meta:
        if not np.array_equal(dense_meta[name], sparse_meta[name]):
            raise AssertionError(f"5s/60s {name} arrays are not aligned")

    labels = dense_meta["labels"]
    users = dense_meta["user_ids"]
    pairs = dense_meta["pair_ids"]
    train_users, dev_users = select_dev_users(
        labels, users, args.seed, args.dev_ratio, args.trials)
    train_indices = np.flatnonzero(np.isin(users, train_users))
    dev_indices = np.flatnonzero(np.isin(users, dev_users))
    if set(map(int, train_users)) & set(map(int, dev_users)):
        raise AssertionError("Train-A and Dev-A users overlap")

    split_manifest = {
        "protocol": "geolife-v12-user-disjoint-dev-v1",
        "seed": args.seed,
        "source_split": "existing five-rate training users only",
        "formal_validation_or_test_loaded": False,
        "train_a": {
            "users": list(map(int, train_users)),
            "user_count": int(len(train_users)),
            "window_count": int(len(train_indices)),
            "class_counts": dict(Counter(map(str, labels[train_indices]))),
        },
        "dev_a": {
            "users": list(map(int, dev_users)),
            "user_count": int(len(dev_users)),
            "window_count": int(len(dev_indices)),
            "class_counts": dict(Counter(map(str, labels[dev_indices]))),
        },
        "checks": {
            "train_dev_user_disjoint": True,
            "all_five_classes_in_train_a": bool(np.all(class_counts(labels[train_indices]) > 0)),
            "all_five_classes_in_dev_a": bool(np.all(class_counts(labels[dev_indices]) > 0)),
            "source_users_are_original_training_users": True,
        },
    }

    split_results = {}
    split_results["train_a"], train_valid_values = _build_split(
        "train_a", train_indices, dense_trjs, sparse_trjs,
        labels, users, pairs, output_dir)
    split_results["dev_a"], _ = _build_split(
        "dev_a", dev_indices, dense_trjs, sparse_trjs,
        labels, users, pairs, output_dir)
    train_mean = train_valid_values.mean(axis=0)
    train_std = train_valid_values.std(axis=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    np.savez(
        output_dir / "train_normalization.npz",
        target_names=np.asarray(TARGET_NAMES),
        mean=train_mean,
        std=train_std,
    )

    target_manifest = {
        "protocol": "geolife-v12-dense-behavior-targets-v1",
        "dense_condition": dense_condition,
        "sparse_condition": sparse_condition,
        "source": "original five-rate train split only",
        "formal_validation_or_test_loaded": False,
        "target_names": list(TARGET_NAMES),
        "sparse_input_names": list(INPUT_NAMES),
        "stop_distance_threshold_meters": STOP_DISTANCE_METERS,
        "interval_alignment": "dense edge start >= sparse t_i and dense edge end <= sparse t_(i+1)",
        "interpolation": False,
        "normalization_fit_split": "train_a only",
        "train_target_mean": dict(zip(TARGET_NAMES, map(float, train_mean))),
        "train_target_std": dict(zip(TARGET_NAMES, map(float, train_std))),
        "splits": split_results,
        "checks": {
            "five_and_sixty_second_metadata_equal": True,
            "timestamp_based_alignment": True,
            "invalid_intervals_marked": True,
            "dev_statistics_not_used_for_normalization": True,
            "formal_test_not_loaded": True,
        },
    }
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.target_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    args.target_manifest.write_text(
        json.dumps(target_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("V12 Checkpoint A target build: PASSED")
    print(f"Train-A: {len(train_users)} users, {len(train_indices)} windows")
    print(f"Dev-A: {len(dev_users)} users, {len(dev_indices)} windows")
    for name, result in split_results.items():
        print(
            f"{name}: {result['valid_dense_target_intervals']}/"
            f"{result['sparse_interval_count']} valid intervals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--views-root", type=Path,
        default=Path("data/geolife_five_rate_views"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/v12_behavior_targets"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/manifests/v12_dev_split.json"))
    parser.add_argument(
        "--target-manifest", type=Path,
        default=Path("reports/manifests/v12_behavior_target_manifest.json"))
    parser.add_argument("--seed", type=int, default=12013)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=20_000)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
