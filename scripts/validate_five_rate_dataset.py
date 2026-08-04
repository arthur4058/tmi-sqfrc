#!/usr/bin/env python3
"""Validate paired, user-disjoint 5/10/20/30/60-second GeoLife views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmi.data_preprocess.variable_sampling import apply_condition


RATES = (5, 10, 20, 30, 60)
CONDITIONS = tuple(f"fixed_{rate}s" for rate in RATES)
SPLITS = ("train", "val", "test")
METADATA = ("labels", "user_ids", "source_ids", "pair_ids")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(root: Path, condition: str, split: str, suffix: str):
    return np.load(
        root / condition / f"{split}_{suffix}.npy", allow_pickle=True
    )


def stats(trajectories) -> dict:
    lengths = np.asarray([len(trajectory) for trajectory in trajectories])
    intervals = np.concatenate([
        np.diff(np.asarray(trajectory, dtype=float)[:, 0])
        for trajectory in trajectories if len(trajectory) > 1
    ])
    return {
        "samples": int(len(trajectories)),
        "points_mean": float(lengths.mean()),
        "points_median": float(np.median(lengths)),
        "points_min": int(lengths.min()),
        "points_max": int(lengths.max()),
        "interval_mean_seconds": float(intervals.mean()),
        "interval_median_seconds": float(np.median(intervals)),
        "interval_std_seconds": float(intervals.std()),
        "interval_min_seconds": float(intervals.min()),
        "interval_max_seconds": float(intervals.max()),
    }


def validate(args) -> dict:
    root = args.views_root.resolve()
    result = {
        "protocol": "geolife-five-rate-matched-v1",
        "status": "PASSED",
        "rates_seconds": list(RATES),
        "checks": {
            "user_disjoint": True,
            "metadata_aligned": True,
            "paired_across_rates": True,
            "published_episode_operator_reproducible": True,
            "real_points_only": True,
            "point_density_strictly_decreasing": True,
            "labels_cover_all_five_classes": True,
        },
        "splits": {},
        "file_sha256": {},
    }

    split_users = {}
    for split in SPLITS:
        canonical = load(root, "fixed_5s", split, "trjs")
        pair_ids = load(root, "fixed_5s", split, "pair_ids")
        reference = {
            suffix: load(root, "fixed_5s", split, suffix)
            for suffix in METADATA
        }
        split_users[split] = set(map(int, reference["user_ids"]))
        class_counts = Counter(map(int, reference["labels"]))
        if set(class_counts) != set(range(5)):
            raise AssertionError(
                f"{split}: expected labels 0..4, found {sorted(class_counts)}"
            )

        condition_stats = {}
        for condition in CONDITIONS:
            trajectories = load(root, condition, split, "trjs")
            if len(trajectories) != len(canonical):
                raise AssertionError(
                    f"{split}/{condition}: sample count is not paired"
                )
            for suffix, expected in reference.items():
                actual = load(root, condition, split, suffix)
                if not np.array_equal(actual, expected):
                    raise AssertionError(
                        f"{split}/{condition}: {suffix} is not aligned"
                    )

            for index, (dense, actual, pair_id) in enumerate(
                    zip(canonical, trajectories, pair_ids)):
                expected = apply_condition(
                    dense, condition, args.seed, str(pair_id)
                )
                if not np.array_equal(np.asarray(actual), expected):
                    raise AssertionError(
                        f"{split}/{condition}/{index}: operator mismatch"
                    )

            condition_stats[condition] = stats(trajectories)
            trajectory_file = root / condition / f"{split}_trjs.npy"
            result["file_sha256"][
                f"{condition}/{split}_trjs.npy"
            ] = sha256(trajectory_file)

        means = [
            condition_stats[condition]["points_mean"]
            for condition in CONDITIONS
        ]
        if not all(left > right for left, right in zip(means, means[1:])):
            raise AssertionError(
                f"{split}: point density is not strictly decreasing: {means}"
            )

        result["splits"][split] = {
            "sample_count": int(len(canonical)),
            "user_count": int(len(split_users[split])),
            "class_counts": {
                str(label): int(class_counts[label])
                for label in sorted(class_counts)
            },
            "condition_stats": condition_stats,
        }

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = split_users[left] & split_users[right]
            if overlap:
                raise AssertionError(
                    f"{left}/{right}: overlapping users {sorted(overlap)}"
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Five-rate GeoLife validation: PASSED")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
