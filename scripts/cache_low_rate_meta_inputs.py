"""Cache frozen low-rate expert features for fast meta-model experiments."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.low_rate_user_stacker import collect_experts
from scripts.train_multiscale_short_official import load_split
from scripts.train_multiscale_short_trajectory import ROOT


def cache(seed=10086, device="cuda"):
    output = ROOT / f"experiments/geolife_meta_cache_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        features, labels, base = collect_experts(split, seed, device)
        users = load_split(split)[-1]
        np.savez_compressed(
            output / f"{split}.npz",
            features=features,
            labels=labels,
            base=base,
            users=users,
        )
        print(split, features.shape, labels.shape, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache(args.seed, args.device)
