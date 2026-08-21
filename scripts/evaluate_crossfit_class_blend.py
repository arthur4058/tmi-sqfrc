"""Validation-selected robust blend of V2 and the class-aware stacker.

The stacker is kept as a five-fold ensemble instead of refitting one model on
all validation samples.  A second cross-fitting layer learns only five class
blend coefficients, choosing whether each class should rely more on the
stable V2 consensus or on the class-aware stacker.  Selection and test remain
separate commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from scripts.evaluate_class_aware_stacker import (
    collect_three_logits,
    fit_linear_stacker,
    predict_from_parameters,
    probability_features,
)
from scripts.evaluate_dual_expert_consensus import (
    choose_temperature,
    metrics,
    softmax,
)
from scripts.evaluate_three_expert_consensus import choose_simplex_mix


ROOT = Path(__file__).resolve().parents[1]
FOLDS = 5
C_VALUE = 0.01
BALANCE_POWER = 0.25
ALPHA_GRID = np.linspace(0.0, 1.0, 9)


def blend_by_class(stable_probability, adaptive_probability, alphas):
    """Blend corresponding class scores and restore probability sums."""
    alphas = np.asarray(alphas, dtype=np.float64).reshape(1, -1)
    mixed = ((1.0 - alphas) * stable_probability
             + alphas * adaptive_probability)
    return mixed / mixed.sum(axis=1, keepdims=True)


def select_class_alphas(
        stable_probability, adaptive_probability, targets,
        grid=ALPHA_GRID, rounds=3, shrinkage=0.001):
    """Coordinate search for five low-capacity, mildly regularized gates."""
    num_classes = stable_probability.shape[1]
    alphas = np.full(num_classes, 0.5, dtype=np.float64)
    for _ in range(rounds):
        changed = False
        for class_index in range(num_classes):
            best = None
            for value in grid:
                candidate_alphas = alphas.copy()
                candidate_alphas[class_index] = value
                probability = blend_by_class(
                    stable_probability, adaptive_probability,
                    candidate_alphas,
                )
                score = metrics(probability, targets)
                penalty = shrinkage * np.square(
                    candidate_alphas - 0.5
                ).mean()
                rank = (
                    score["accuracy"] + score["macro_f1"] - penalty,
                    min(score["accuracy"], score["macro_f1"]),
                    -float(np.square(candidate_alphas - 0.5).mean()),
                )
                if best is None or rank > best[0]:
                    best = (rank, value)
            if best[1] != alphas[class_index]:
                changed = True
                alphas[class_index] = best[1]
        if not changed:
            break
    return alphas


def build_crossfit_stacker(features, targets, folds=FOLDS):
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=10086
    )
    oof = np.zeros((targets.size, int(targets.max()) + 1), dtype=np.float64)
    parameters = []
    for train_indices, held_out_indices in splitter.split(features, targets):
        model = fit_linear_stacker(
            features[train_indices], targets[train_indices],
            C_VALUE, BALANCE_POWER,
        )
        oof[held_out_indices] = model.predict_proba(
            features[held_out_indices]
        )
        parameters.append({
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        })
    return oof, parameters


def crossfit_class_blend(stable_probability, adaptive_probability, targets):
    splitter = StratifiedKFold(
        n_splits=FOLDS, shuffle=True, random_state=20260821
    )
    oof = np.zeros_like(stable_probability, dtype=np.float64)
    fold_alphas = []
    for train_indices, held_out_indices in splitter.split(
            stable_probability, targets):
        alphas = select_class_alphas(
            stable_probability[train_indices],
            adaptive_probability[train_indices],
            targets[train_indices],
        )
        fold_alphas.append(alphas.tolist())
        oof[held_out_indices] = blend_by_class(
            stable_probability[held_out_indices],
            adaptive_probability[held_out_indices], alphas,
        )
    final_alphas = select_class_alphas(
        stable_probability, adaptive_probability, targets
    )
    return oof, fold_alphas, final_alphas


def ensemble_predict(features, parameters):
    predictions = [
        predict_from_parameters(
            features, item["coefficients"], item["intercept"]
        )
        for item in parameters
    ]
    return np.mean(predictions, axis=0)


def select(rate, artifact_path):
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    logits, targets = collect_three_logits(
        rate, "validation", artifact_path.parent
    )
    temperatures = [choose_temperature(item, targets) for item in logits]
    probabilities = [
        softmax(item, temperature)
        for item, temperature in zip(logits, temperatures)
    ]
    v2_weights = choose_simplex_mix(probabilities, targets)
    v2_probability = sum(
        weight * probability
        for weight, probability in zip(v2_weights, probabilities)
    )
    features = probability_features(probabilities)
    stack_oof, stack_parameters = build_crossfit_stacker(features, targets)
    blend_oof, fold_alphas, final_alphas = crossfit_class_blend(
        v2_probability, stack_oof, targets
    )
    result = {
        "protocol": "geolife-crossfit-class-blend-v10",
        "sampling_interval_seconds": rate,
        "selection_split": "user-disjoint-validation",
        "test_touched_during_selection": False,
        "temperatures": [float(value) for value in temperatures],
        "v2_weights": [float(value) for value in v2_weights],
        "stacker": {
            "folds": FOLDS,
            "C": C_VALUE,
            "balance_power": BALANCE_POWER,
            "parameters": stack_parameters,
        },
        "class_blend": {
            "fold_alphas": fold_alphas,
            "final_alphas": final_alphas.tolist(),
        },
        "validation": {
            "v2": metrics(v2_probability, targets),
            "stacker_oof": metrics(stack_oof, targets),
            "v10_nested_oof": metrics(blend_oof, targets),
        },
    }
    artifact_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def evaluate_test(rate, artifact_path, result_path):
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact["sampling_interval_seconds"] != rate:
        raise ValueError("artifact sampling interval mismatch")
    if artifact["test_touched_during_selection"] is not False:
        raise ValueError("invalid selection provenance")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    logits, targets = collect_three_logits(rate, "test", result_path.parent)
    probabilities = [
        softmax(item, temperature)
        for item, temperature in zip(logits, artifact["temperatures"])
    ]
    v2_probability = sum(
        weight * probability
        for weight, probability in zip(
            artifact["v2_weights"], probabilities
        )
    )
    stack_probability = ensemble_predict(
        probability_features(probabilities),
        artifact["stacker"]["parameters"],
    )
    v10_probability = blend_by_class(
        v2_probability, stack_probability,
        artifact["class_blend"]["final_alphas"],
    )
    result = {
        "protocol": artifact["protocol"],
        "sampling_interval_seconds": rate,
        "selection_artifact": str(artifact_path),
        "test_fit_performed": False,
        "test_samples": int(targets.size),
        "baseline": metrics(probabilities[0], targets),
        "v2_consensus": metrics(v2_probability, targets),
        "crossfit_stacker": metrics(stack_probability, targets),
        "v10": metrics(v10_probability, targets),
    }
    result["delta_vs_baseline"] = {
        name: result["v10"][name] - result["baseline"][name]
        for name in ("accuracy", "macro_f1")
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "test"))
    parser.add_argument("--rate", type=int, choices=(30, 60), default=60)
    parser.add_argument(
        "--artifact",
        default="experiments/geolife_crossfit_class_blend_v10_60s_seed10086/selection.json",
    )
    parser.add_argument(
        "--result",
        default="experiments/geolife_crossfit_class_blend_v10_60s_seed10086/test_results.json",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    artifact_path = ROOT / args.artifact
    if args.command == "select":
        select(args.rate, artifact_path)
    else:
        evaluate_test(args.rate, artifact_path, ROOT / args.result)


if __name__ == "__main__":
    main()
