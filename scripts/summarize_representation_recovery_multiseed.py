"""Aggregate frozen one-shot recovery/V2 results across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def aggregate(seeds):
    rows = []
    for seed in seeds:
        path = (
            ROOT / f"experiments/geolife_recovery_v2_fusion_seed{seed}"
            / "formal_test_results.json"
        )
        result = json.loads(path.read_text(encoding="utf-8"))["test"]
        rows.append({
            "seed": seed,
            "v2": result["v2"],
            "fusion": result["fusion"],
            "delta_vs_v2": result["fusion_delta_vs_v2"],
        })

    metrics = ("accuracy", "macro_f1")
    aggregate_metrics = {}
    for metric in metrics:
        aggregate_metrics[metric] = {
            "v2": summary([row["v2"][metric] for row in rows]),
            "fusion": summary([row["fusion"][metric] for row in rows]),
            "delta_vs_v2": summary([
                row["delta_vs_v2"][metric] for row in rows
            ]),
            "positive_seeds": sum(
                row["delta_vs_v2"][metric] > 0.0 for row in rows
            ),
        }
    result = {
        "protocol": "geolife-low-rate-representation-recovery-v1-multiseed",
        "sampling_interval_seconds": 60,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate_metrics,
        "decision": "not_yet_stable_enough_for_primary_claim",
    }
    output = (
        ROOT / "reports/experiments/"
        "geolife_low_rate_representation_recovery_v1_multiseed.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[10086, 42, 2024])
    args = parser.parse_args()
    aggregate(args.seeds)


if __name__ == "__main__":
    main()
