"""Class-aware stacking of the three frozen low-rate experts.

Selection and final evaluation are deliberately separate commands.  The
selection command only iterates over the user-disjoint validation loader and
stores the frozen calibrator.  The evaluation command then applies that
artifact to the independent test loader without fitting anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import main as training_main
from scripts.evaluate_dual_expert_consensus import (
    choose_temperature,
    collect_logits,
    metrics,
    softmax,
)
from scripts.evaluate_motion_expert_consensus import collect_expert_logits
from scripts.evaluate_three_expert_consensus import (
    choose_simplex_mix,
    load_motion_expert,
)
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]
EXPERT_NAMES = ("baseline", "four_feature_expert", "motion_expert")


def probability_features(probabilities):
    """Return finite log-probability features for the linear stacker."""
    return np.concatenate([
        np.log(np.clip(probability, 1e-8, 1.0))
        for probability in probabilities
    ], axis=1)


def class_power_weights(targets, power):
    """Mild minority weighting; power=1 is fully balanced, 0 is unweighted."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("class-balance power must be in [0, 1]")
    counts = np.bincount(targets)
    class_weights = (targets.size / (len(counts) * counts)) ** power
    return class_weights[targets]


def fit_linear_stacker(features, targets, c_value, balance_power):
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
        random_state=10086,
    )
    model.fit(
        features,
        targets,
        sample_weight=class_power_weights(targets, balance_power),
    )
    return model


def predict_from_parameters(features, coefficients, intercept):
    logits = features @ np.asarray(coefficients).T + np.asarray(intercept)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def cross_validated_stacker(
        features,
        targets,
        c_values=(0.003, 0.01, 0.03, 0.1),
        balance_powers=(0.0, 0.25, 0.5),
        folds=5):
    """Select a small stacker using only out-of-fold validation predictions."""
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=10086
    )
    trials = []
    best = None
    num_classes = int(np.max(targets)) + 1
    for c_value in c_values:
        for balance_power in balance_powers:
            oof_probability = np.zeros(
                (targets.size, num_classes), dtype=np.float64)
            for train_indices, held_out_indices in splitter.split(features, targets):
                model = fit_linear_stacker(
                    features[train_indices], targets[train_indices],
                    c_value, balance_power,
                )
                oof_probability[held_out_indices] = model.predict_proba(
                    features[held_out_indices]
                )
            trial_metrics = metrics(oof_probability, targets)
            trial = {
                "C": float(c_value),
                "balance_power": float(balance_power),
                "accuracy": trial_metrics["accuracy"],
                "macro_f1": trial_metrics["macro_f1"],
            }
            trials.append(trial)
            # Accuracy and macro-F1 have equal importance.  The final two
            # terms prefer the simpler and less reweighted solution on ties.
            rank = (
                trial_metrics["accuracy"] + trial_metrics["macro_f1"],
                min(trial_metrics["accuracy"], trial_metrics["macro_f1"]),
                -float(balance_power),
                -float(c_value),
            )
            if best is None or rank > best[0]:
                best = (rank, trial)
    selected = best[1]
    final_model = fit_linear_stacker(
        features, targets, selected["C"], selected["balance_power"]
    )
    return selected, trials, final_model


def setup_models(rate, output_dir):
    baseline_path = ROOT / (
        f"experiments/geolife_matched_b0_fixed_{rate}s_seed10086/"
        "matched_test/configuration.json"
    )
    old_dir = (
        ROOT / "experiments/geolife_60s_b2_feature_seed10086"
        if rate == 60
        else ROOT / f"experiments/geolife_feature_expert_{rate}s_seed10086"
    )
    motion_dir = ROOT / f"experiments/geolife_motion_expert_{rate}s_seed10086"

    b0_config = json.loads(baseline_path.read_text(encoding="utf-8"))
    b0_config.update({
        "output_dir": str(output_dir),
        "records_file": str(output_dir / "records.xlsx"),
    })
    dataset.config = b0_config
    pipeline = training_main.TrainingPipeline(b0_config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()

    old_config = json.loads(
        (old_dir / "configuration.json").read_text(encoding="utf-8")
    )
    old_config["test_only"] = None
    old_model = model_factory(old_config, pipeline.train_data.feature_data)
    old_checkpoint = torch.load(
        old_dir / "checkpoints/model_best.pth",
        map_location="cpu", weights_only=True,
    )
    old_model.load_state_dict(old_checkpoint["state_dict"], strict=True)
    old_model.to(pipeline.device)

    _, motion_pipeline, motion_model = load_motion_expert(
        motion_dir, b0_config
    )
    return (
        b0_config, pipeline, old_model,
        pipeline.train_data.feature_data.max_seq_len,
        motion_pipeline, motion_model,
    )


def collect_three_logits(rate, split, output_dir):
    (
        b0_config, pipeline, old_model, old_max_len,
        motion_pipeline, motion_model,
    ) = setup_models(rate, output_dir)
    if split == "validation":
        base_loader = pipeline.val_loader
        motion_loader = motion_pipeline.val_loader
    elif split == "test":
        base_loader = pipeline.test_loader
        motion_loader = motion_pipeline.test_loader
    else:
        raise ValueError(f"unsupported split: {split}")

    b0_logits, old_logits, targets = collect_logits(
        pipeline.model, old_model, base_loader,
        pipeline.device, b0_config["seed"], old_max_len,
    )
    motion_logits, motion_targets = collect_expert_logits(
        motion_model, motion_loader, motion_pipeline.device
    )
    if not np.array_equal(targets, motion_targets):
        raise ValueError(f"{split} targets are not aligned between experts")
    return [b0_logits, old_logits, motion_logits], targets


def select(rate, artifact_path):
    output_dir = artifact_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logits, targets = collect_three_logits(rate, "validation", output_dir)
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
    selected, trials, model = cross_validated_stacker(features, targets)
    fitted_probability = model.predict_proba(features)
    result = {
        "protocol": "geolife-class-aware-stacking-v8",
        "sampling_interval_seconds": rate,
        "selection_split": "user-disjoint-validation",
        "test_touched_during_selection": False,
        "experts": list(EXPERT_NAMES),
        "temperatures": [float(value) for value in temperatures],
        "v2_weights": [float(value) for value in v2_weights],
        "v2_validation": metrics(v2_probability, targets),
        "cross_validation_selected": selected,
        "cross_validation_trials": trials,
        "refit_validation": metrics(fitted_probability, targets),
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    artifact_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def evaluate_test(rate, artifact_path, result_path):
    calibration = json.loads(artifact_path.read_text(encoding="utf-8"))
    if calibration["sampling_interval_seconds"] != rate:
        raise ValueError("calibration artifact sampling interval mismatch")
    if not calibration["test_touched_during_selection"] is False:
        raise ValueError("invalid calibration provenance")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    logits, targets = collect_three_logits(rate, "test", result_path.parent)
    probabilities = [
        softmax(item, temperature)
        for item, temperature in zip(logits, calibration["temperatures"])
    ]
    v2_probability = sum(
        weight * probability
        for weight, probability in zip(
            calibration["v2_weights"], probabilities
        )
    )
    stack_probability = predict_from_parameters(
        probability_features(probabilities),
        calibration["coefficients"], calibration["intercept"],
    )
    result = {
        "protocol": calibration["protocol"],
        "sampling_interval_seconds": rate,
        "calibration_artifact": str(artifact_path),
        "test_fit_performed": False,
        "test_samples": int(targets.size),
        "baseline": metrics(probabilities[0], targets),
        "v2_consensus": metrics(v2_probability, targets),
        "class_aware_stacker": metrics(stack_probability, targets),
    }
    result["delta_vs_baseline"] = {
        name: result["class_aware_stacker"][name]
        - result["baseline"][name]
        for name in ("accuracy", "macro_f1")
    }
    result["delta_vs_v2"] = {
        name: result["class_aware_stacker"][name]
        - result["v2_consensus"][name]
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
        default="experiments/geolife_class_aware_stacker_v8_60s_seed10086/calibration.json",
    )
    parser.add_argument(
        "--result",
        default="experiments/geolife_class_aware_stacker_v8_60s_seed10086/test_results.json",
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
