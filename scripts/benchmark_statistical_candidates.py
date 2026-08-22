"""Exploratory cross-split audit of location-invariant statistical models.

This script is diagnostic only: the test split has already been inspected and
must not be used to select a paper result.  It determines whether a whole model
family is worth taking to a fresh confirmation protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.evaluate_dual_expert_consensus import metrics, softmax
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_statistical_motion_expert import (
    candidates,
    load_split,
    load_v2_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def run(seed: int = 10086):
    os.chdir(ROOT)
    output_dir = ROOT / f"experiments/geolife_statistical_audit_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_split("train", seed, training=True)
    val_x, val_y = load_split("val", seed, training=False)
    test_x, test_y = load_split("test", seed, training=False)
    v2_val, val_targets, temperatures, weights = load_v2_validation(seed)
    logits, test_targets = load_v2_test(seed, output_dir)
    if not np.array_equal(val_y, val_targets):
        raise ValueError("validation targets are not aligned")
    if not np.array_equal(test_y, test_targets):
        raise ValueError("test targets are not aligned")
    v2_test = sum(
        weight * softmax(logit, temperature)
        for weight, logit, temperature in zip(weights, logits, temperatures)
    )

    result = {
        "protocol": "exploratory-statistical-family-audit-v1",
        "not_for_model_selection": True,
        "location_and_timestamp_invariant": True,
        "seed": seed,
        "v2": {"validation": metrics(v2_val, val_y),
               "test": metrics(v2_test, test_y)},
        "candidates": [],
    }
    validation = json.loads((
        ROOT / f"experiments/geolife_statistical_motion_seed{seed}/validation.json"
    ).read_text(encoding="utf-8"))
    trial_by_name = {trial["name"]: trial for trial in validation["trials"]}

    for name, model in candidates(seed).items():
        model.fit(train_x, train_y)
        val_raw = model.predict_proba(val_x)
        test_raw = model.predict_proba(test_x)
        trial = trial_by_name[name]
        temperature = trial["temperature"]
        weight = trial["fusion_weight"]
        val_probability = softmax(np.log(val_raw.clip(1e-9)), temperature)
        test_probability = softmax(np.log(test_raw.clip(1e-9)), temperature)
        val_fusion = (1.0 - weight) * v2_val + weight * val_probability
        test_fusion = (1.0 - weight) * v2_test + weight * test_probability
        test_score = metrics(test_fusion, test_y)
        base_test = result["v2"]["test"]
        result["candidates"].append({
            "name": name,
            "validation_weight": weight,
            "standalone": {
                "validation": metrics(val_probability, val_y),
                "test": metrics(test_probability, test_y),
            },
            "fusion": {
                "validation": metrics(val_fusion, val_y),
                "test": test_score,
                "test_delta_vs_v2": {
                    key: test_score[key] - base_test[key]
                    for key in ("accuracy", "macro_f1")
                },
            },
        })

    path = output_dir / "audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
