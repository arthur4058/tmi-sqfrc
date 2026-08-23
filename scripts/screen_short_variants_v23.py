"""Screen V23 variants on validation, then optionally test the frozen winner."""

from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.evaluate_dual_expert_consensus import choose_temperature, metrics, softmax
from scripts.evaluate_multiscale_short_fusion import collect_short_probability
from scripts.evaluate_user_robust_v2_v16 import choose_user_robust_weight
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_multiscale_short_official import load_split
from scripts.train_multiscale_short_trajectory import ROOT
from scripts.train_statistical_motion_expert import load_v2_validation


def validation(seed, device):
    root = ROOT / f"experiments/geolife_short_variants_v23_seed{seed}"
    v2, labels, temperatures, v2_weights = load_v2_validation(seed)
    users = load_split("val")[-1]
    trials = []
    for checkpoint in sorted(root.glob("*/fold0_hybrid_best.pth")):
        logits, candidate_labels = collect_short_probability(
            "val", seed, checkpoint, 512, device
        )
        if not np.array_equal(labels, candidate_labels):
            raise ValueError("Validation labels are not aligned")
        temperature = choose_temperature(logits, labels)
        probability = softmax(logits, temperature)
        selected, _ = choose_user_robust_weight(v2, probability, labels, users)
        trials.append({
            "name": checkpoint.parent.name,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "temperature": temperature,
            "standalone": metrics(probability, labels),
            "selected": selected,
        })
    if not trials:
        raise RuntimeError(f"No checkpoints found under {root}")
    winner = max(trials, key=lambda item: (
        item["selected"]["users_both_nonnegative"],
        item["selected"]["q25_joint_delta"],
        min(item["selected"]["global_delta"].values()),
    ))
    result = {
        "protocol": "geolife-v23-short-variant-screen",
        "seed": seed,
        "selection_split": "user-disjoint validation only",
        "v2_validation": metrics(v2, labels),
        "winner": winner,
        "trials": trials,
        "v2_temperatures": temperatures,
        "v2_weights": np.asarray(v2_weights, dtype=float).tolist(),
    }
    (root / "screen.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def frozen_test(result, device):
    seed = result["seed"]
    winner = result["winner"]
    root = ROOT / f"experiments/geolife_short_variants_v23_seed{seed}"
    v2_logits, labels = load_v2_test(seed, root)
    v2 = sum(
        weight * softmax(logits, temperature)
        for weight, logits, temperature in zip(
            result["v2_weights"], v2_logits, result["v2_temperatures"]
        )
    )
    short_logits, short_labels = collect_short_probability(
        "test", seed, ROOT / winner["checkpoint"], 512, device
    )
    if not np.array_equal(labels, short_labels):
        raise ValueError("Test labels are not aligned")
    short = softmax(short_logits, winner["temperature"])
    weight = winner["selected"]["weight"]
    fused = (1.0 - weight) * v2 + weight * short
    base = metrics(v2, labels)
    score = metrics(fused, labels)
    result["test"] = {
        "v2": base,
        "winner_standalone": metrics(short, labels),
        "fusion": score,
        "delta": {key: score[key] - base[key] for key in ("accuracy", "macro_f1")},
    }
    (root / "frozen_test.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-winner", action="store_true")
    args = parser.parse_args()
    result = validation(args.seed, args.device)
    if args.test_winner:
        result = frozen_test(result, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
