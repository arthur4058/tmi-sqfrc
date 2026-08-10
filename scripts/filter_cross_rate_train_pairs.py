#!/usr/bin/env python3
"""Keep only sparse training segments with a valid dense teacher window."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARRAY_FILES = (
    "clean_trj_segs.npy",
    "noise_trj_segs.npy",
    "trj_seg_masks.npy",
    "clean_multi_feature_segs.npy",
    "noise_multi_feature_segs.npy",
    "fs_seg_masks.npy",
    "clean_multi_feature_seg_labels.npy",
    "noise_multi_feature_seg_labels.npy",
    "segment_pair_ids.npy",
    "segment_time_ranges.npy",
)


def main():
    repo = Path(__file__).resolve().parents[1]
    teacher_dir = repo / "data/geolife_v3_fixed_5s_features/train"
    teacher_pairs = set(map(str, np.load(
        teacher_dir / "segment_pair_ids.npy", allow_pickle=True)))
    report = {
        "policy": "retain sparse train segments whose physical window exists at 5s",
        "teacher_rate_seconds": 5,
        "rates": {},
    }

    for rate in (20, 60):
        base = repo / f"data/geolife_v3_fixed_{rate}s_features/train"
        pair_ids = np.load(base / "segment_pair_ids.npy", allow_pickle=True)
        keep = np.fromiter(
            (str(pair_id) in teacher_pairs for pair_id in pair_ids),
            dtype=bool,
            count=len(pair_ids),
        )
        before = int(len(pair_ids))
        for filename in ARRAY_FILES:
            path = base / filename
            values = np.load(path, allow_pickle=True)
            if len(values) != before:
                raise ValueError(
                    f"Length mismatch before filtering: {rate}s {filename}")
            np.save(path, values[keep])
        after = int(keep.sum())
        report["rates"][str(rate)] = {
            "segments_before": before,
            "segments_after": after,
            "segments_removed": before - after,
            "retained_fraction": after / before,
        }
        print(
            f"{rate}s train: retained {after}/{before} segments "
            f"({after / before:.2%})"
        )

    output = repo / "reports/manifests/geolife_cross_rate_filter_v3.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
