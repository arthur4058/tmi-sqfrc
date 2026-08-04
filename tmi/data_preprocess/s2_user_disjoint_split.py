"""Deterministic user-disjoint GeoLife train/validation/test split."""

import argparse
import json
import os
from collections import Counter

import numpy as np
from logzero import logger


SPLITS = ("train", "val", "test")


def _class_counts(labels, n_classes=5):
    return np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)


def find_user_split(labels, user_ids, ratios=(0.7, 0.1, 0.2), seed=42,
                    trials=20000, n_classes=5):
    """Search deterministic user permutations for a class-balanced split."""
    labels = np.asarray(labels, dtype=int)
    user_ids = np.asarray(user_ids)
    unique_users = np.unique(user_ids)
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")

    n_train = int(round(len(unique_users) * ratios[0]))
    n_val = int(round(len(unique_users) * ratios[1]))
    n_train = min(max(n_train, 1), len(unique_users) - 2)
    n_val = min(max(n_val, 1), len(unique_users) - n_train - 1)

    user_counts = {
        int(user): _class_counts(labels[user_ids == user], n_classes)
        for user in unique_users
    }
    global_dist = _class_counts(labels, n_classes).astype(float)
    global_dist /= global_dist.sum()
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(trials):
        perm = rng.permutation(unique_users)
        groups = (
            np.sort(perm[:n_train]),
            np.sort(perm[n_train:n_train + n_val]),
            np.sort(perm[n_train + n_val:]),
        )
        counts = [
            np.sum([user_counts[int(user)] for user in group], axis=0)
            for group in groups
        ]
        if any(np.any(count == 0) for count in counts):
            continue
        score = 0.0
        for count in counts:
            split_dist = count / count.sum()
            score += float(np.mean(np.abs(split_dist - global_dist)))
        candidate = (score, groups, counts)
        if best is None or score < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError(
            "Could not find a user-disjoint split containing every class")
    return {
        split: np.asarray(users)
        for split, users in zip(SPLITS, best[1])
    }


def validate_user_disjoint_split(split_users, labels=None, user_ids=None,
                                 n_classes=5):
    user_sets = {name: set(map(int, users)) for name, users in split_users.items()}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = user_sets[left] & user_sets[right]
            if overlap:
                raise AssertionError(f"{left}/{right} user overlap: {sorted(overlap)}")
    if labels is not None and user_ids is not None:
        for split in SPLITS:
            mask = np.isin(user_ids, list(user_sets[split]))
            if np.any(_class_counts(labels[mask], n_classes) == 0):
                raise AssertionError(f"{split} does not contain all {n_classes} classes")
    return True


def split_and_save(data_dir, save_dir, manifest_path, ratios, seed, trials):
    arrays = {
        "trjs": np.load(os.path.join(data_dir, "trjs.npy"), allow_pickle=True),
        "labels": np.load(os.path.join(data_dir, "labels.npy")),
        "user_ids": np.load(os.path.join(data_dir, "user_ids.npy")),
        "source_ids": np.load(os.path.join(data_dir, "source_ids.npy")),
    }
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"unaligned extracted arrays: {lengths}")

    split_users = find_user_split(
        arrays["labels"], arrays["user_ids"], ratios, seed, trials)
    validate_user_disjoint_split(
        split_users, arrays["labels"], arrays["user_ids"])
    os.makedirs(save_dir, exist_ok=True)

    manifest = {
        "protocol": "geolife-user-disjoint-v1",
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "source_sample_count": len(arrays["labels"]),
        "splits": {},
    }
    for split in SPLITS:
        users = split_users[split]
        mask = np.isin(arrays["user_ids"], users)
        for name, values in arrays.items():
            np.save(os.path.join(save_dir, f"{split}_{name}.npy"), values[mask])
        split_labels = arrays["labels"][mask]
        manifest["splits"][split] = {
            "users": list(map(int, users)),
            "user_count": len(users),
            "sample_count": int(mask.sum()),
            "class_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(map(int, split_labels)).items())
            },
        }
        logger.info(
            "%s: %d users, %d samples, classes=%s",
            split, len(users), mask.sum(),
            manifest["splits"][split]["class_counts"])

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()
    ratios = tuple(float(value) for value in args.ratios.split(","))
    if len(ratios) != 3:
        raise ValueError("--ratios must contain train,val,test")
    split_and_save(
        args.data_dir, args.save_dir, args.manifest_path,
        ratios, args.seed, args.trials)


if __name__ == "__main__":
    main()
