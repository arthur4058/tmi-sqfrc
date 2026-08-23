"""Fuse V2 and V16 using a validation-user robustness constraint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_dual_expert_consensus import (
    choose_temperature,
    metrics,
    softmax,
)
from scripts.evaluate_multiscale_short_fusion import collect_short_probability
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_multiscale_short_official import load_split
from scripts.train_multiscale_short_trajectory import ROOT
from scripts.train_statistical_motion_expert import load_v2_validation


def choose_user_robust_weight(v2, short, labels, users):
    base_global = metrics(v2, labels)
    best = None
    trials = []
    for weight in np.linspace(0.0, 0.5, 51):
        fused = (1.0 - weight) * v2 + weight * short
        global_score = metrics(fused, labels)
        user_deltas = []
        for user in np.unique(users):
            index = users == user
            base = metrics(v2[index], labels[index])
            score = metrics(fused[index], labels[index])
            user_deltas.append([
                score["accuracy"] - base["accuracy"],
                score["macro_f1"] - base["macro_f1"],
            ])
        values = np.asarray(user_deltas)
        joint = values.min(axis=1)
        delta_accuracy = global_score["accuracy"] - base_global["accuracy"]
        delta_f1 = global_score["macro_f1"] - base_global["macro_f1"]
        trial = {
            "weight": float(weight),
            "global": global_score,
            "global_delta": {
                "accuracy": delta_accuracy,
                "macro_f1": delta_f1,
            },
            "users_both_nonnegative": float((joint >= -1e-12).mean()),
            "q25_joint_delta": float(np.quantile(joint, 0.25)),
            "median_joint_delta": float(np.median(joint)),
        }
        trials.append(trial)
        key = (
            trial["users_both_nonnegative"],
            trial["q25_joint_delta"],
            min(delta_accuracy, delta_f1),
            -float(weight),
        )
        if best is None or key > best[0]:
            best = (key, trial)
    return best[1], trials


def evaluate(seed: int, device: str = "cuda"):
    output = ROOT / f"experiments/geolife_multiscale_short_official_seed{seed}"
    checkpoint = output / "fold0_hybrid_best.pth"
    short_val_logits, val_labels = collect_short_probability(
        "val", seed, checkpoint, 512, device
    )
    v2_val, v2_val_labels, temperatures, v2_weights = load_v2_validation(seed)
    users = load_split("val")[-1]
    if not np.array_equal(val_labels, v2_val_labels):
        raise ValueError("Validation labels are not aligned")
    short_temperature = choose_temperature(short_val_logits, val_labels)
    short_val = softmax(short_val_logits, short_temperature)
    selected, trials = choose_user_robust_weight(
        v2_val, short_val, val_labels, users
    )
    weight = selected["weight"]

    # Test is loaded only after the rule and weight are fixed on validation users.
    v2_logits, test_labels = load_v2_test(seed, output)
    v2_test = sum(
        model_weight * softmax(logits, temperature)
        for model_weight, logits, temperature
        in zip(v2_weights, v2_logits, temperatures)
    )
    short_test_logits, short_test_labels = collect_short_probability(
        "test", seed, checkpoint, 512, device
    )
    if not np.array_equal(test_labels, short_test_labels):
        raise ValueError("Test labels are not aligned")
    short_test = softmax(short_test_logits, short_temperature)
    fused_test = (1.0 - weight) * v2_test + weight * short_test
    v2_score = metrics(v2_test, test_labels)
    fusion_score = metrics(fused_test, test_labels)
    result = {
        "protocol": "geolife-v2-v16-user-robust-weight-v21",
        "seed": seed,
        "selection_split": "user-disjoint validation",
        "selection_rule": (
            "maximize fraction of validation users non-degrading on both metrics, "
            "then lower-quartile joint gain"
        ),
        "test_touched_during_selection": False,
        "short_temperature": short_temperature,
        "selected": selected,
        "test": {
            "v2": v2_score,
            "fusion": fusion_score,
            "delta": {
                key: fusion_score[key] - v2_score[key]
                for key in ("accuracy", "macro_f1")
            },
        },
        "validation_trials": trials,
    }
    target = output / "v2_v16_user_robust_weight_v21.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.seed, args.device)


if __name__ == "__main__":
    main()
