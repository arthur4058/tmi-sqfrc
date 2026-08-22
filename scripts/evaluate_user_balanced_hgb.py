"""Train CV-selected user-balanced HGB and evaluate its frozen fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.cross_validate_statistical_experts_by_user import load_training
from scripts.cross_validate_user_balanced_hgb import user_weights
from scripts.evaluate_dual_expert_consensus import choose_temperature, metrics, softmax
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_statistical_motion_expert import (
    choose_mix,
    load_split,
    load_v2_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def run(seed=10086, power=0.5):
    clean, noise, labels, users = load_training()
    model = HistGradientBoostingClassifier(
        learning_rate=0.07, max_iter=180, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight="balanced", random_state=seed,
    )
    weights = user_weights(users, power)
    model.fit(np.concatenate([clean, noise]), np.concatenate([labels, labels]),
              sample_weight=np.concatenate([weights, weights]))

    val_x, val_y = load_split("val", seed, training=False)
    raw_val = model.predict_proba(val_x)
    temperature = choose_temperature(np.log(raw_val.clip(1e-9)), val_y)
    expert_val = softmax(np.log(raw_val.clip(1e-9)), temperature)
    v2_val, targets, v2_temperatures, v2_weights = load_v2_validation(seed)
    if not np.array_equal(val_y, targets):
        raise ValueError("validation targets are not aligned")
    fusion_weight, v2_score, fusion_score = choose_mix(v2_val, expert_val, val_y)
    delta = {key: fusion_score[key] - v2_score[key]
             for key in ("accuracy", "macro_f1")}
    accepted = delta["accuracy"] >= 0.01 and delta["macro_f1"] >= 0.01
    result = {
        "protocol": "geolife-user-balanced-hgb-frozen-v1",
        "seed": seed,
        "user_weight_power_selected_by_training_cv": power,
        "temperature": temperature,
        "fusion_weight": fusion_weight,
        "validation": {
            "v2": v2_score,
            "expert": metrics(expert_val, val_y),
            "fusion": fusion_score,
            "delta_vs_v2": delta,
        },
        "validation_gate_passed": accepted,
    }
    if accepted:
        output = ROOT / f"experiments/geolife_user_balanced_hgb_seed{seed}"
        logits, test_y = load_v2_test(seed, output)
        v2_test = sum(
            weight * softmax(logit, temp)
            for weight, logit, temp in zip(v2_weights, logits, v2_temperatures)
        )
        test_x, statistical_targets = load_split("test", seed, training=False)
        if not np.array_equal(test_y, statistical_targets):
            raise ValueError("test targets are not aligned")
        raw_test = model.predict_proba(test_x)
        expert_test = softmax(np.log(raw_test.clip(1e-9)), temperature)
        fusion_test = (1.0 - fusion_weight) * v2_test + fusion_weight * expert_test
        base = metrics(v2_test, test_y)
        fused = metrics(fusion_test, test_y)
        result["confirmation"] = {
            "parameters_frozen_before_evaluation": True,
            "v2": base,
            "expert": metrics(expert_test, test_y),
            "fusion": fused,
            "delta_vs_v2": {key: fused[key] - base[key]
                            for key in ("accuracy", "macro_f1")},
        }
    output = ROOT / f"experiments/geolife_user_balanced_hgb_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--power", type=float, default=0.5)
    args = parser.parse_args()
    run(args.seed, args.power)


if __name__ == "__main__":
    main()
