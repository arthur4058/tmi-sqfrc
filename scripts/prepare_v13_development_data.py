#!/usr/bin/env python3
"""Prepare leakage-free 60-second Train-A/Dev-A arrays for V13."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("data/geolife_five_rate_views/fixed_60s"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/manifests/v12_dev_split.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/geolife_v13_dev_split_60s"))
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    arrays = {
        name: np.load(args.source / f"train_{name}.npy", allow_pickle=True)
        for name in ("trjs", "labels", "user_ids", "source_ids", "pair_ids")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    split_users = {
        "train": manifest["train_a"]["users"],
        "val": manifest["dev_a"]["users"],
        "test": manifest["dev_a"]["users"],
    }
    for split, users in split_users.items():
        mask = np.isin(arrays["user_ids"], users)
        for name, values in arrays.items():
            np.save(args.output / f"{split}_{name}.npy", values[mask])
        print(
            f"{split}: users={len(np.unique(arrays['user_ids'][mask]))} "
            f"windows={int(mask.sum())}")


if __name__ == "__main__":
    main()
