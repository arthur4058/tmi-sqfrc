"""Run the one-shot formal test with frozen validation calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import main as training_main
from scripts.evaluate_dual_expert_consensus import collect_logits, metrics, softmax
from scripts.evaluate_motion_expert_consensus import (
    collect_baseline_logits,
    collect_expert_logits,
)
from scripts.evaluate_three_expert_consensus import load_motion_expert
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]


def load_v2_test(seed):
    baseline_dir = ROOT / f"experiments/geolife_matched_b0_fixed_60s_seed{seed}"
    four_dir = ROOT / f"experiments/geolife_60s_b2_feature_seed{seed}"
    motion_dir = ROOT / f"experiments/geolife_motion_expert_60s_seed{seed}"
    output_dir = ROOT / f"experiments/geolife_v6_1_v2_fusion_seed{seed}"

    config = json.loads(
        (baseline_dir / "configuration.json").read_text(encoding="utf-8")
    )
    config.update({
        "task": "dual_branch_classification",
        "test_only": "testset",
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
        pipeline.model, four_model, pipeline.test_loader,
        pipeline.device, seed, four_max_len,
    )
    motion_logits, motion_targets = collect_expert_logits(
        motion_model, motion_pipeline.test_loader, motion_pipeline.device,
    )
    if not np.array_equal(targets, motion_targets):
        raise ValueError("V2 test targets are not aligned")
    return [b0_logits, four_logits, motion_logits], targets


def load_v6_1_test(seed):
    experiment = ROOT / f"experiments/geolife_v6_1_balanced_paired_60s_seed{seed}"
    config = json.loads(
        (experiment / "configuration.json").read_text(encoding="utf-8")
    )
    config.update({
        "task": "dual_branch_classification",
        "test_only": "testset",
        "paired_multirate_consistency": False,
        "load_model": str(experiment / "checkpoints/model_best.pth"),
    })
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()
    return collect_baseline_logits(
        pipeline.model, pipeline.test_loader, pipeline.device,
    )


def formal_test(seed=10086):
    output_dir = ROOT / f"experiments/geolife_v6_1_v2_fusion_seed{seed}"
    calibration_path = output_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("accepted_for_formal_test"):
        raise ValueError("Validation gate did not authorize formal testing")
    if calibration.get("formal_test_used"):
        raise ValueError("Formal test was already recorded for this calibration")

    v2_logits, targets = load_v2_test(seed)
    v61_logits, v61_targets = load_v6_1_test(seed)
    if not np.array_equal(targets, v61_targets):
        raise ValueError("V2 and V6.1 test targets are not aligned")

    v2_probabilities = [
        softmax(logits, temperature)
        for logits, temperature in zip(
            v2_logits, calibration["temperatures"]["v2_experts"]
        )
    ]
    v2_probability = sum(
        weight * probability
        for weight, probability in zip(
            calibration["v2_weights"], v2_probabilities
        )
    )
    v61_probability = softmax(
        v61_logits, calibration["temperatures"]["v6_1"]
    )
    residual_weight = calibration["v6_1_residual_weight"]
    fusion_probability = (
        (1.0 - residual_weight) * v2_probability
        + residual_weight * v61_probability
    )

    v2_score = metrics(v2_probability, targets)
    v61_score = metrics(v61_probability, targets)
    fusion_score = metrics(fusion_probability, targets)
    result = {
        "protocol": "geolife-v6.1-balanced-paired-v2-formal-test",
        "sampling_interval_seconds": 60,
        "seed": seed,
        "test_samples": int(len(targets)),
        "calibration_source": str(calibration_path.relative_to(ROOT)),
        "parameters_frozen_on_validation": True,
        "test": {
            "v2": v2_score,
            "v6_1": v61_score,
            "fusion": fusion_score,
            "fusion_delta_vs_v2": {
                name: fusion_score[name] - v2_score[name]
                for name in ("accuracy", "macro_f1")
            },
        },
    }
    (output_dir / "formal_test_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    calibration["formal_test_used"] = True
    calibration["formal_test_results"] = "formal_test_results.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    os.chdir(ROOT)
    formal_test(args.seed)


if __name__ == "__main__":
    main()
