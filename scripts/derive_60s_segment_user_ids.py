"""Reproduce S3/S4 ordering and recover segment user IDs for any fixed rate."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from geopy.distance import geodesic

from tmi.data_preprocess.s3_data_augmentation import TrajectoryAugmentation


ROOT = Path(__file__).resolve().parents[1]
MIN_POINTS = 4
MAX_POINTS = 200
MAX_STAY = 1200
SPEED_LIMIT = {0: 7, 1: 12, 2: 120 / 3.6, 3: 180 / 3.6, 4: 120 / 3.6}
ACC_LIMIT = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}


def replay_augmentation(trjs, labels, users, seed=42):
    np.random.seed(seed)
    output_trjs, output_labels, output_users = [], [], []
    counts = Counter(map(int, labels))
    target = max(counts.values())
    methods = [
        TrajectoryAugmentation.reverse,
        TrajectoryAugmentation.random_rotate,
        TrajectoryAugmentation.spatial_translate,
        TrajectoryAugmentation.random_crop,
    ]
    for label in counts:
        indices = np.flatnonzero(labels == label)
        class_trjs = [trjs[index] for index in indices]
        class_users = [users[index] for index in indices]
        output_trjs.extend(class_trjs)
        output_labels.extend([label] * len(indices))
        output_users.extend(class_users)
        for _ in range(target - len(indices)):
            index = np.random.randint(len(indices))
            augmented = deepcopy(class_trjs[index])
            n_methods = np.random.randint(1, len(methods) + 1)
            selected = np.random.choice(methods, n_methods, replace=False)
            for method in selected:
                augmented = method(augmented)
            output_trjs.append(augmented)
            output_labels.append(label)
            output_users.append(class_users[index])
    return (np.asarray(output_trjs, dtype=object),
            np.asarray(output_labels), np.asarray(output_users))


def filter_trajectory(trajectory):
    invalid = []
    for index in range(len(trajectory) - 1):
        point_a = trajectory[index - 1] if index in invalid else trajectory[index]
        point_b = trajectory[index + 1]
        delta = point_b[0] - point_a[0]
        if delta <= 0:
            invalid.append(index + 1)
            continue
        for point_index, point in ((index, point_a), (index + 1, point_b)):
            lat, lon = point[1], point[2]
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                invalid.append(point_index)
    invalid_indices = np.unique(np.asarray(invalid, dtype=np.int64))
    filtered = np.delete(trajectory, invalid_indices, axis=0)
    return filtered if len(filtered) >= MIN_POINTS else None


def split_trajectory(trajectory):
    split_at = np.flatnonzero(np.diff(trajectory[:, 0]) > MAX_STAY) + 1
    pieces = np.split(trajectory, split_at)
    output = []
    for piece in pieces:
        for start in range(0, len(piece), MAX_POINTS):
            segment = piece[start:start + MAX_POINTS]
            if len(segment) >= MIN_POINTS:
                output.append(segment)
    return output


def survives_feature_filter(segment, label):
    previous_velocity = 0.0
    valid = 0
    for index in range(len(segment) - 1):
        a, b = segment[index], segment[index + 1]
        delta = b[0] - a[0]
        distance = geodesic([a[1], a[2]], [b[1], b[2]]).meters
        velocity = distance / delta
        acceleration = (velocity - previous_velocity) / delta
        if abs(velocity) <= SPEED_LIMIT[int(label)] and abs(acceleration) <= ACC_LIMIT[int(label)]:
            valid += 1
            previous_velocity = velocity
    return valid >= MIN_POINTS


def derive(trjs, labels, users, seed=42):
    order = np.random.RandomState(seed).permutation(len(trjs))
    segment_labels, segment_users = [], []
    for index in order:
        trajectory = filter_trajectory(trjs[index])
        if trajectory is None:
            continue
        for segment in split_trajectory(trajectory):
            if survives_feature_filter(segment, labels[index]):
                segment_labels.append(int(labels[index]))
                segment_users.append(int(users[index]))
    return np.asarray(segment_labels), np.asarray(segment_users, dtype=np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=int, choices=(5, 10, 20, 30, 60), default=60)
    args = parser.parse_args()
    views = ROOT / f"data/geolife_five_rate_views/fixed_{args.rate}s"
    augmented = ROOT / f"data/geolife_five_rate_{args.rate}s_augmented"
    features = ROOT / f"data/geolife_five_rate_fixed_{args.rate}s_features"

    train_trjs = np.load(views / "train_trjs.npy", allow_pickle=True)
    train_labels = np.load(views / "train_labels.npy")
    train_users = np.load(views / "train_user_ids.npy")
    replay_trjs, replay_labels, replay_users = replay_augmentation(
        train_trjs, train_labels, train_users, args.seed)
    stored_trjs = np.load(augmented / "train_trjs_augmented.npy", allow_pickle=True)
    stored_labels = np.load(augmented / "train_labels_augmented.npy")
    if not np.array_equal(replay_labels, stored_labels):
        raise ValueError("S3 label replay does not match stored augmentation")
    if len(replay_trjs) != len(stored_trjs) or any(
            not np.array_equal(left, right)
            for left, right in zip(replay_trjs, stored_trjs)):
        raise ValueError("S3 random replay does not match stored trajectories")

    inputs = {
        "train": (stored_trjs, stored_labels, replay_users),
        "val": tuple(np.load(views / f"val_{name}.npy", allow_pickle=True)
                     for name in ("trjs", "labels", "user_ids")),
        "test": tuple(np.load(views / f"test_{name}.npy", allow_pickle=True)
                      for name in ("trjs", "labels", "user_ids")),
    }
    for split, arrays in inputs.items():
        labels, users = derive(*arrays, seed=args.seed)
        expected = np.load(features / split / "clean_multi_feature_seg_labels.npy")
        if not np.array_equal(labels, expected):
            raise ValueError(f"{split} derived labels do not align with S4 output")
        np.save(features / split / "segment_user_ids.npy", users)
        print(split, len(users), len(np.unique(users)))


if __name__ == "__main__":
    main()
