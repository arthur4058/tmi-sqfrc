"""Calibrate the representation-recovery expert with V2 on validation only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import main as training_main
from scripts.evaluate_dual_expert_consensus import (
    choose_temperature,
    collect_logits,
    metrics,
    softmax,
)
from scripts.evaluate_motion_expert_consensus import (
    collect_baseline_logits,
    collect_expert_logits,
)
from scripts.evaluate_three_expert_consensus import (
    choose_simplex_mix,
    load_motion_expert,
)
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]


def choose_residual_mix(base_probability, candidate_probability, targets,
                        maximum_weight=0.5, step=0.01):
    base_metrics = metrics(base_probability, targets)
    best = None
    for weight in np.arange(0.0, maximum_weight + step / 2.0, step):
        probability = (
            (1.0 - weight) * base_probability
            + weight * candidate_probability
        )
        score = metrics(probability, targets)
        delta_accuracy = score["accuracy"] - base_metrics["accuracy"]
        delta_f1 = score["macro_f1"] - base_metrics["macro_f1"]
        candidate = (
            min(delta_accuracy, delta_f1),
            delta_accuracy + delta_f1,
            -float(weight),
            float(weight),
            score,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3], base_metrics, best[4]


def load_v2_validation(seed):
    baseline_dir = ROOT / f"experiments/geolife_matched_b0_fixed_60s_seed{seed}"
    four_dir = ROOT / f"experiments/geolife_60s_b2_feature_seed{seed}"
    motion_dir = ROOT / f"experiments/geolife_motion_expert_60s_seed{seed}"
    output_dir = ROOT / f"experiments/geolife_recovery_v2_fusion_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        (baseline_dir / "configuration.json").read_text(encoding="utf-8")
    )
    config.update({
        "task": "dual_branch_classification",
        "test_only": None,
        "load_model": str(baseline_dir / "checkpoints/model_best.pth"),
        "output_dir": str(output_dir),
        "records_file": str(output_dir / "records.xlsx"),
    })
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()

    four_config = json.loads(
        (four_dir / "configuration.json").read_text(encoding="utf-8")
    )
    four_config["test_only"] = None
    four_model = model_factory(four_config, pipeline.train_data.feature_data)
    checkpoint = torch.load(
        four_dir / "checkpoints/model_best.pth",
        map_location="cpu", weights_only=True,
    )
    four_model.load_state_dict(checkpoint["state_dict"], strict=True)
    four_model.to(pipeline.device)
    four_max_len = pipeline.train_data.feature_data.max_seq_len

    _, motion_pipeline, motion_model = load_motion_expert(motion_dir, config)
    b0_logits, four_logits, targets = collect_logits(
        pipeline.model, four_model, pipeline.val_loader,
        pipeline.device, seed, four_max_len,
    )
    motion_logits, motion_targets = collect_expert_logits(
        motion_model, motion_pipeline.val_loader, motion_pipeline.device,
    )
    if not np.array_equal(targets, motion_targets):
        raise ValueError("V2 validation targets are not aligned")
    return [b0_logits, four_logits, motion_logits], targets


def load_recovery_validation(seed):
    experiment = ROOT / f"experiments/geolife_representation_recovery_60s_seed{seed}"
    config = json.loads(
        (experiment / "configuration.json").read_text(encoding="utf-8")
    )
    config.update({
        "task": "dual_branch_classification",
        "test_only": None,
        "paired_multirate_consistency": False,
        "load_model": str(experiment / "checkpoints/model_best.pth"),
    })
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()
    return collect_baseline_logits(
        pipeline.model, pipeline.val_loader, pipeline.device,
    )


def calibrate(seed=10086, minimum_accuracy_gain=0.003,
              minimum_f1_gain=0.005):
    v2_logits, targets = load_v2_validation(seed)
    recovery_logits, recovery_targets = load_recovery_validation(seed)
    if not np.array_equal(targets, recovery_targets):
        raise ValueError("V2 and recovery validation targets are not aligned")

    v2_temperatures = [
        choose_temperature(logits, targets) for logits in v2_logits
    ]
    v2_probabilities = [
        softmax(logits, temperature)
        for logits, temperature in zip(v2_logits, v2_temperatures)
    ]
    v2_weights = choose_simplex_mix(v2_probabilities, targets)
    v2_probability = sum(
        weight * probability
        for weight, probability in zip(v2_weights, v2_probabilities)
    )
    recovery_temperature = choose_temperature(recovery_logits, targets)
    recovery_probability = softmax(recovery_logits, recovery_temperature)
    residual_weight, v2_score, fusion_score = choose_residual_mix(
        v2_probability, recovery_probability, targets,
    )
    delta = {
        name: fusion_score[name] - v2_score[name]
        for name in ("accuracy", "macro_f1")
    }
    accepted = (
        delta["accuracy"] >= minimum_accuracy_gain
        and delta["macro_f1"] >= minimum_f1_gain
    )
    result = {
        "protocol": "geolife-representation-recovery-v2-validation",
        "sampling_interval_seconds": 60,
        "seed": seed,
        "selection_split": "user-disjoint-validation",
        "formal_test_used": False,
        "temperatures": {
            "v2_experts": v2_temperatures,
            "recovery": recovery_temperature,
        },
        "v2_weights": [float(value) for value in v2_weights],
        "recovery_residual_weight": residual_weight,
        "validation": {
            "v2": v2_score,
            "recovery": metrics(recovery_probability, targets),
            "fusion": fusion_score,
            "fusion_delta_vs_v2": delta,
        },
        "acceptance_threshold": {
            "accuracy": minimum_accuracy_gain,
            "macro_f1": minimum_f1_gain,
        },
        "accepted_for_formal_test": accepted,
    }
    output = ROOT / f"experiments/geolife_recovery_v2_fusion_seed{seed}/calibration.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    os.chdir(ROOT)
    calibrate(args.seed)


if __name__ == "__main__":
    main()
