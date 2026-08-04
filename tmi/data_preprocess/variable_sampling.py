"""Literature-backed physical-time sampling views for GeoLife.

Fixed-rate views use the equal-duration episode operator described by
Burkhard et al. (2020): split time into equal episodes and retain the first
real observation in every non-empty episode.  The variable-rate view keeps
the same operator but changes the episode length between consecutive
physical-time blocks.  Coordinates are never interpolated or synthesized.
"""

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
from logzero import logger


DEFAULT_CONDITIONS = (
    "fixed_5s",
    "fixed_30s",
    "fixed_60s",
    "variable_5_60s",
)

VARIABLE_INTERVALS_SECONDS = (5, 10, 15, 30, 60)
VARIABLE_BLOCK_SECONDS = 60


def clean_trajectory(trajectory):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape (n_points, 3)")
    trajectory = trajectory[np.all(np.isfinite(trajectory), axis=1)]
    trajectory = trajectory[np.argsort(trajectory[:, 0], kind="stable")]
    if len(trajectory):
        _, first = np.unique(trajectory[:, 0], return_index=True)
        trajectory = trajectory[np.sort(first)]
    return trajectory


def physical_windows(trajectory, window_seconds=300, stride_seconds=150):
    """Return complete physical-time windows without interpolation."""
    trajectory = clean_trajectory(trajectory)
    if len(trajectory) < 2:
        return []
    first_time = trajectory[0, 0]
    last_time = trajectory[-1, 0]
    if last_time - first_time < window_seconds:
        return []
    starts = np.arange(
        first_time, last_time - window_seconds + 1e-9, stride_seconds)
    windows = []
    for start in starts:
        end = start + window_seconds
        mask = (trajectory[:, 0] >= start) & (trajectory[:, 0] <= end)
        window = trajectory[mask]
        if len(window) >= 2:
            windows.append((float(start), window))
    return windows


def fixed_interval_sample(trajectory, interval_seconds, anchor_time=None):
    """Keep the first real observation in each equal-duration episode.

    Empty episodes remain empty.  This is equivalent to the temporal
    subsampling operator used by Burkhard et al. (2020), and it deliberately
    avoids coordinate interpolation.
    """
    trajectory = clean_trajectory(trajectory)
    if len(trajectory) < 2 or interval_seconds <= 0:
        return trajectory
    anchor = trajectory[0, 0] if anchor_time is None else float(anchor_time)
    episode_ids = np.floor(
        (trajectory[:, 0] - anchor) / float(interval_seconds) + 1e-12
    ).astype(np.int64)
    _, first_indices = np.unique(episode_ids, return_index=True)
    return trajectory[np.sort(first_indices)]


def piecewise_variable_sample(
        trajectory, rng, intervals=VARIABLE_INTERVALS_SECONDS,
        block_seconds=VARIABLE_BLOCK_SECONDS):
    """Apply a deterministic piecewise-variable episode schedule.

    For a 300-second window, a seeded permutation assigns 5, 10, 15, 30 and
    60-second episode lengths to consecutive 60-second blocks.  This keeps
    all observations real while ensuring that every window contains multiple
    acquisition rates.  It is an explicit extension of the published fixed
    episode operator, not a missing-point corruption model.
    """
    trajectory = clean_trajectory(trajectory)
    if len(trajectory) < 2:
        return trajectory
    intervals = tuple(int(value) for value in intervals)
    if not intervals or min(intervals) <= 0 or block_seconds <= 0:
        raise ValueError("intervals and block_seconds must be positive")

    start = float(trajectory[0, 0])
    block_ids = np.floor(
        (trajectory[:, 0] - start) / float(block_seconds) + 1e-12
    ).astype(np.int64)
    schedule = rng.permutation(np.asarray(intervals, dtype=int))
    sampled_blocks = []
    for block_id in np.unique(block_ids):
        block = trajectory[block_ids == block_id]
        interval = int(schedule[int(block_id) % len(schedule)])
        sampled_blocks.append(
            fixed_interval_sample(
                block, interval,
                anchor_time=start + int(block_id) * block_seconds,
            )
        )
    return clean_trajectory(np.concatenate(sampled_blocks, axis=0))


def deterministic_rng(seed, pair_id, condition):
    material = f"{seed}|{pair_id}|{condition}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


def apply_condition(trajectory, condition, seed, pair_id, min_points=5):
    if condition.startswith("fixed_"):
        interval = int(condition.removeprefix("fixed_").removesuffix("s"))
        sampled = fixed_interval_sample(trajectory, interval)
    elif condition == "variable_5_60s":
        sampled = piecewise_variable_sample(
            trajectory, deterministic_rng(seed, pair_id, condition))
    else:
        raise ValueError(f"unknown sampling condition: {condition}")
    return clean_trajectory(sampled)


def build_paired_views(trjs, labels, user_ids, source_ids, conditions,
                       seed=42, window_seconds=300, stride_seconds=150,
                       min_points=5):
    arrays = (trjs, labels, user_ids, source_ids)
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("input arrays are not aligned")
    records = {condition: [] for condition in conditions}
    metadata = []
    rejected = Counter()

    for trajectory, label, user_id, source_id in zip(*arrays):
        for window_index, (start, window) in enumerate(
                physical_windows(trajectory, window_seconds, stride_seconds)):
            pair_id = f"{source_id}:w{window_index}:{int(start)}"
            # All experimental conditions are subsets of one canonical 5 s
            # view. This prevents dense raw windows from being split into
            # extra model samples and makes comparisons genuinely paired.
            canonical = fixed_interval_sample(window, 5)
            views = {
                condition: apply_condition(
                    canonical, condition, seed, pair_id, min_points)
                for condition in conditions
            }
            invalid = [
                condition for condition, view in views.items()
                if len(view) < min_points
            ]
            if invalid:
                for condition in invalid:
                    rejected[condition] += 1
                continue
            for condition, view in views.items():
                records[condition].append(view)
            metadata.append((int(label), int(user_id), str(source_id), pair_id))

    if not metadata:
        raise RuntimeError("no physical windows survived all sampling conditions")
    labels_out, users_out, sources_out, pairs_out = map(np.asarray, zip(*metadata))
    views_out = {
        condition: np.asarray(values, dtype=object)
        for condition, values in records.items()
    }
    return views_out, labels_out, users_out, sources_out, pairs_out, rejected


def _point_stats(trjs):
    lengths = np.asarray([len(trajectory) for trajectory in trjs])
    intervals = np.concatenate([
        np.diff(np.asarray(trajectory)[:, 0])
        for trajectory in trjs if len(trajectory) > 1
    ])
    return {
        "points_mean": float(lengths.mean()),
        "points_median": float(np.median(lengths)),
        "points_min": int(lengths.min()),
        "points_max": int(lengths.max()),
        "interval_mean_seconds": float(np.mean(intervals)),
        "interval_median_seconds": float(np.median(intervals)),
        "interval_std_seconds": float(np.std(intervals)),
        "interval_max_seconds": float(np.max(intervals)),
    }


def build_and_save(data_dir, output_dir, manifest_path, conditions,
                   seed, window_seconds, stride_seconds, min_points):
    manifest = {
        "protocol": "geolife-published-episode-sampling-v2",
        "sampling_operator": {
            "name": "equal-duration episodes, first real observation",
            "reference": (
                "Burkhard et al. (2020), Transportation Research Part C, "
                "doi:10.1016/j.trc.2020.01.021"
            ),
            "interpolation": False,
        },
        "variable_view": {
            "name": "piecewise-variable extension",
            "intervals_seconds": list(VARIABLE_INTERVALS_SECONDS),
            "block_seconds": VARIABLE_BLOCK_SECONDS,
            "schedule": "seeded permutation per paired window",
        },
        "seed": seed,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "minimum_points": min_points,
        "paired_across_conditions": True,
        "conditions": list(conditions),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        inputs = [
            np.load(os.path.join(data_dir, f"{split}_{suffix}.npy"),
                    allow_pickle=True)
            for suffix in ("trjs", "labels", "user_ids", "source_ids")
        ]
        views, labels, users, sources, pairs, rejected = build_paired_views(
            *inputs, conditions, seed, window_seconds, stride_seconds, min_points)
        manifest["splits"][split] = {
            "sample_count": len(labels),
            "user_count": len(np.unique(users)),
            "class_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(map(int, labels)).items())
            },
            "rejected_windows": dict(rejected),
            "view_stats": {},
        }
        for condition, trjs in views.items():
            condition_dir = os.path.join(output_dir, condition)
            os.makedirs(condition_dir, exist_ok=True)
            for suffix, values in (
                ("trjs", trjs),
                ("labels", labels),
                ("user_ids", users),
                ("source_ids", sources),
                ("pair_ids", pairs),
            ):
                np.save(
                    os.path.join(condition_dir, f"{split}_{suffix}.npy"), values)
            manifest["splits"][split]["view_stats"][condition] = _point_stats(trjs)
        logger.info(
            "%s: %d paired windows, %d users, classes=%s",
            split, len(labels), len(np.unique(users)),
            manifest["splits"][split]["class_counts"])

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window_seconds", type=int, default=300)
    parser.add_argument("--stride_seconds", type=int, default=150)
    parser.add_argument("--min_points", type=int, default=5)
    args = parser.parse_args()
    build_and_save(
        args.data_dir, args.output_dir, args.manifest_path,
        tuple(args.conditions.split(",")), args.seed,
        args.window_seconds, args.stride_seconds, args.min_points)


if __name__ == "__main__":
    main()
