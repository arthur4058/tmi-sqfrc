"""Validation-user robust convex fusion of V2 and two V23 short experts."""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from scripts.evaluate_dual_expert_consensus import choose_temperature, metrics, softmax
from scripts.evaluate_multiscale_short_fusion import collect_short_probability
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_multiscale_short_official import load_split
from scripts.train_multiscale_short_trajectory import ROOT
from scripts.train_statistical_motion_expert import load_v2_validation


def user_robust_score(base, fused, labels, users):
    base_global = metrics(base, labels)
    score = metrics(fused, labels)
    joint = []
    for user in np.unique(users):
        index = users == user
        before = metrics(base[index], labels[index])
        after = metrics(fused[index], labels[index])
        joint.append(min(
            after["accuracy"] - before["accuracy"],
            after["macro_f1"] - before["macro_f1"],
        ))
    delta = {
        key: score[key] - base_global[key]
        for key in ("accuracy", "macro_f1")
    }
    return {
        "global": score,
        "global_delta": delta,
        "users_both_nonnegative": float((np.asarray(joint) >= -1e-12).mean()),
        "q25_joint_delta": float(np.quantile(joint, 0.25)),
        "median_joint_delta": float(np.median(joint)),
    }


def rank(item):
    score = item["score"]
    return (
        score["users_both_nonnegative"],
        score["q25_joint_delta"],
        min(score["global_delta"].values()),
        sum(score["global_delta"].values()),
        -(item["weight_a"] + item["weight_b"]),
    )


def evaluate(seed=10086, device="cuda", test_winner=False):
    root = ROOT / f"experiments/geolife_short_variants_v23_seed{seed}"
    v2, labels, v2_temperatures, v2_weights = load_v2_validation(seed)
    users = load_split("val")[-1]
    experts = {}
    for checkpoint in sorted(root.glob("*/fold0_hybrid_best.pth")):
        logits, candidate_labels = collect_short_probability(
            "val", seed, checkpoint, 512, device
        )
        if not np.array_equal(labels, candidate_labels):
            raise ValueError("Validation labels are not aligned")
        temperature = choose_temperature(logits, labels)
        experts[checkpoint.parent.name] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "temperature": temperature,
            "validation": softmax(logits, temperature),
        }

    trials = []
    for name_a, name_b in itertools.combinations(experts, 2):
        for weight_a in np.linspace(0.0, 0.4, 21):
            for weight_b in np.linspace(0.0, 0.4, 21):
                if weight_a + weight_b > 0.55 or weight_a + weight_b == 0:
                    continue
                fused = (
                    (1.0 - weight_a - weight_b) * v2
                    + weight_a * experts[name_a]["validation"]
                    + weight_b * experts[name_b]["validation"]
                )
                trials.append({
                    "expert_a": name_a,
                    "expert_b": name_b,
                    "weight_a": float(weight_a),
                    "weight_b": float(weight_b),
                    "score": user_robust_score(v2, fused, labels, users),
                })
    winner = max(trials, key=rank)
    result = {
        "protocol": "geolife-v2-two-short-experts-v24",
        "seed": seed,
        "selection_split": "user-disjoint validation only",
        "winner": winner,
        "experts": {
            name: {key: value for key, value in item.items() if key != "validation"}
            for name, item in experts.items()
        },
        "v2_temperatures": v2_temperatures,
        "v2_weights": np.asarray(v2_weights, dtype=float).tolist(),
    }
    if test_winner:
        v2_logits, test_labels = load_v2_test(seed, root)
        v2_test = sum(
            weight * softmax(logits, temperature)
            for weight, logits, temperature in zip(
                result["v2_weights"], v2_logits, v2_temperatures
            )
        )
        short_test = {}
        for name in (winner["expert_a"], winner["expert_b"]):
            item = experts[name]
            logits, candidate_labels = collect_short_probability(
                "test", seed, ROOT / item["checkpoint"], 512, device
            )
            if not np.array_equal(test_labels, candidate_labels):
                raise ValueError("Test labels are not aligned")
            short_test[name] = softmax(logits, item["temperature"])
        weight_a, weight_b = winner["weight_a"], winner["weight_b"]
        fused = (
            (1.0 - weight_a - weight_b) * v2_test
            + weight_a * short_test[winner["expert_a"]]
            + weight_b * short_test[winner["expert_b"]]
        )
        base, score = metrics(v2_test, test_labels), metrics(fused, test_labels)
        result["test"] = {
            "v2": base,
            "fusion": score,
            "delta": {key: score[key] - base[key] for key in base},
        }
    target = root / ("v24_frozen_test.json" if test_winner else "v24_validation.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "test": result.get("test")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-winner", action="store_true")
    args = parser.parse_args()
    evaluate(args.seed, args.device, args.test_winner)


if __name__ == "__main__":
    main()
