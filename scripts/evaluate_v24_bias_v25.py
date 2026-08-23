"""Validation-only class-boundary calibration for the frozen V24 ensemble."""

from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.evaluate_dual_expert_consensus import metrics, softmax
from scripts.evaluate_multiscale_short_fusion import collect_short_probability
from scripts.evaluate_short_pair_v24 import user_robust_score
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_multiscale_short_official import load_split
from scripts.train_multiscale_short_trajectory import ROOT
from scripts.train_statistical_motion_expert import load_v2_validation


def calibrate(probability, bias):
    logits = np.log(probability.clip(1e-12)) + np.asarray(bias)[None, :]
    return softmax(logits, 1.0)


def candidate_rank(score, bias):
    return (
        score["users_both_nonnegative"],
        score["q25_joint_delta"],
        min(score["global_delta"].values()),
        sum(score["global_delta"].values()),
        -float(np.abs(bias).sum()),
    )


def tune_bias(v2, fusion, labels, users, bias_min=-0.25, bias_max=0.25):
    bias = np.zeros(5, dtype=float)
    best_score = user_robust_score(v2, fusion, labels, users)
    history = []
    for iteration in range(5):
        changed = False
        for class_index in range(1, 5):
            local_best = (candidate_rank(best_score, bias), bias.copy(), best_score)
            for value in np.linspace(bias_min, bias_max, 31):
                proposal = bias.copy()
                proposal[class_index] = value
                score = user_robust_score(
                    v2, calibrate(fusion, proposal), labels, users
                )
                item = (candidate_rank(score, proposal), proposal, score)
                if item[0] > local_best[0]:
                    local_best = item
            if not np.array_equal(local_best[1], bias):
                changed = True
            bias, best_score = local_best[1], local_best[2]
        history.append({
            "iteration": iteration + 1,
            "bias": bias.tolist(),
            "score": best_score,
        })
        if not changed:
            break
    return bias, best_score, history


def collect_v24(split, seed, protocol, device):
    winner = protocol["winner"]
    expert_probabilities = []
    labels = None
    for key in ("expert_a", "expert_b"):
        name = winner[key]
        item = protocol["experts"][name]
        logits, current_labels = collect_short_probability(
            split, seed, ROOT / item["checkpoint"], 512, device
        )
        if labels is not None and not np.array_equal(labels, current_labels):
            raise ValueError("Expert labels are not aligned")
        labels = current_labels
        expert_probabilities.append(softmax(logits, item["temperature"]))
    return expert_probabilities, labels


def evaluate(seed=10086, device="cuda", test_calibration=False, bias_min=-0.25, bias_max=0.25):
    root = ROOT / f"experiments/geolife_short_variants_v23_seed{seed}"
    protocol = json.loads((root / "v24_validation.json").read_text())
    v2_val, labels, v2_temperatures, v2_weights = load_v2_validation(seed)
    experts, expert_labels = collect_v24("val", seed, protocol, device)
    if not np.array_equal(labels, expert_labels):
        raise ValueError("V2 and expert validation labels are not aligned")
    winner = protocol["winner"]
    wa, wb = winner["weight_a"], winner["weight_b"]
    v24_val = (1.0 - wa - wb) * v2_val + wa * experts[0] + wb * experts[1]
    users = load_split("val")[-1]
    bias, score, history = tune_bias(v2_val, v24_val, labels, users, bias_min, bias_max)
    result = {
        "protocol": "geolife-v24-validation-class-bias-v25",
        "seed": seed,
        "selection_split": "user-disjoint validation only",
        "bias_search_range": [bias_min, bias_max],
        "v24": winner,
        "bias": bias.tolist(),
        "validation": {
            "v2": metrics(v2_val, labels),
            "v24": metrics(v24_val, labels),
            "v25": metrics(calibrate(v24_val, bias), labels),
            "robust_score": score,
        },
        "history": history,
    }
    if test_calibration:
        v2_logits, test_labels = load_v2_test(seed, root)
        v2_test = sum(
            weight * softmax(logits, temperature)
            for weight, logits, temperature in zip(
                np.asarray(v2_weights), v2_logits, v2_temperatures
            )
        )
        test_experts, expert_test_labels = collect_v24("test", seed, protocol, device)
        if not np.array_equal(test_labels, expert_test_labels):
            raise ValueError("Test labels are not aligned")
        v24_test = (
            (1.0 - wa - wb) * v2_test
            + wa * test_experts[0] + wb * test_experts[1]
        )
        v25_test = calibrate(v24_test, bias)
        result["test"] = {
            "v2": metrics(v2_test, test_labels),
            "v24": metrics(v24_test, test_labels),
            "v25": metrics(v25_test, test_labels),
        }
    target = root / ("v25_frozen_test.json" if test_calibration else "v25_validation.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-calibration", action="store_true")
    parser.add_argument("--bias-min", type=float, default=-0.25)
    parser.add_argument("--bias-max", type=float, default=0.25)
    args = parser.parse_args()
    evaluate(args.seed, args.device, args.test_calibration, args.bias_min, args.bias_max)


if __name__ == "__main__":
    main()
