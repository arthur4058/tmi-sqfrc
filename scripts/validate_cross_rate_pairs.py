#!/usr/bin/env python3
"""Validate S4 segment-to-physical-window metadata for V3 distillation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    repo = Path(__file__).resolve().parents[1]
    rates = (5, 20, 60)
    splits = ("train", "val", "test")
    report = {"status": "passed", "rates_seconds": list(rates), "splits": {}}
    for split in splits:
        split_report = {}
        reference_pairs = None
        reference_labels = None
        for rate in rates:
            base = repo / f"data/geolife_v3_fixed_{rate}s_features/{split}"
            labels = np.load(base / "noise_multi_feature_seg_labels.npy")
            pair_ids = np.load(base / "segment_pair_ids.npy", allow_pickle=True)
            time_ranges = np.load(base / "segment_time_ranges.npy")
            if not (len(labels) == len(pair_ids) == len(time_ranges)):
                raise ValueError(f"Metadata length mismatch: {rate}s {split}")
            if not np.isfinite(time_ranges).all():
                raise ValueError(f"Non-finite time range: {rate}s {split}")
            if np.any(time_ranges[:, 1] < time_ranges[:, 0]):
                raise ValueError(f"Invalid time range: {rate}s {split}")
            unique_pairs = set(map(str, pair_ids))
            split_report[str(rate)] = {
                "segments": int(len(labels)),
                "unique_physical_windows": int(len(unique_pairs)),
                "class_distribution": {
                    str(int(label)): int(count)
                    for label, count in zip(*np.unique(labels, return_counts=True))
                },
            }
            if rate == 5:
                reference_pairs = unique_pairs
                reference_labels = {}
                for pair_id, label in zip(pair_ids, labels):
                    reference_labels.setdefault(str(pair_id), set()).add(
                        int(label))
            else:
                missing_teacher_pairs = unique_pairs - reference_pairs
                split_report[str(rate)]["missing_5s_teacher_windows"] = len(
                    missing_teacher_pairs)
                if split == "train" and missing_teacher_pairs:
                    raise ValueError(
                        f"{rate}s has {len(missing_teacher_pairs)} windows "
                        "without a 5s teacher segment")
                if split == "train":
                    label_mismatches = sum(
                        int(label) not in reference_labels[str(pair_id)]
                        for pair_id, label in zip(pair_ids, labels)
                    )
                    split_report[str(rate)]["label_mismatches"] = int(
                        label_mismatches)
                    if label_mismatches:
                        raise ValueError(
                            f"{rate}s has {label_mismatches} teacher/student "
                            "label mismatches")
        report["splits"][split] = split_report

    output = repo / "reports/manifests/geolife_cross_rate_pairs_v3.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cross-rate pair validation: PASSED ({output.relative_to(repo)})")


if __name__ == "__main__":
    main()
