#!/usr/bin/env python3
"""Validate alignment and finite values in all five-rate S4 feature sets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


RATES = (5, 10, 20, 30, 60)
SPLITS = ("train", "val", "test")
ARRAYS = (
    "clean_trj_segs",
    "noise_trj_segs",
    "trj_seg_masks",
    "clean_multi_feature_segs",
    "noise_multi_feature_segs",
    "fs_seg_masks",
)


def value_finite(value) -> bool:
    array = np.asarray(value)
    if array.dtype == object:
        return all(value_finite(item) for item in array.flat)
    return bool(np.all(np.isfinite(array.astype(float, copy=False))))


def all_finite(values) -> bool:
    return all(value_finite(value) for value in values)


def validate(repo: Path) -> dict:
    payload = {
        "protocol": "geolife-five-rate-matched-v1",
        "status": "PASSED",
        "checks": {
            "clean_noise_labels_equal": True,
            "all_arrays_aligned": True,
            "all_five_classes_present": True,
            "all_feature_values_finite": True,
        },
        "rates": {},
    }
    for rate in RATES:
        rate_payload = {}
        root = repo / f"data/geolife_five_rate_fixed_{rate}s_features"
        for split in SPLITS:
            clean_labels = np.load(
                root / split / "clean_multi_feature_seg_labels.npy"
            )
            noise_labels = np.load(
                root / split / "noise_multi_feature_seg_labels.npy"
            )
            if not np.array_equal(clean_labels, noise_labels):
                raise AssertionError(f"{rate}s/{split}: clean/noise labels differ")
            counts = Counter(map(int, clean_labels))
            if set(counts) != set(range(5)):
                raise AssertionError(
                    f"{rate}s/{split}: missing classes {sorted(counts)}"
                )

            shapes = {}
            for name in ARRAYS:
                values = np.load(
                    root / split / f"{name}.npy", allow_pickle=True
                )
                if len(values) != len(clean_labels):
                    raise AssertionError(
                        f"{rate}s/{split}/{name}: length {len(values)} != "
                        f"labels {len(clean_labels)}"
                    )
                if not all_finite(values):
                    raise AssertionError(
                        f"{rate}s/{split}/{name}: non-finite values"
                    )
                shapes[name] = list(values.shape)
                del values

            rate_payload[split] = {
                "samples": int(len(clean_labels)),
                "class_counts": {
                    str(label): int(counts[label]) for label in sorted(counts)
                },
                "array_shapes": shapes,
            }
        payload["rates"][f"fixed_{rate}s"] = rate_payload
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/manifests/geolife_five_rate_features_validation.json"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    payload = validate(repo)
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Five-rate S4 feature validation: PASSED")
    print(f"Saved: {output.relative_to(repo)}")


if __name__ == "__main__":
    main()
