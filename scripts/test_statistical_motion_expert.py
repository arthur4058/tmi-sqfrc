"""One-shot test for the validation-frozen statistical motion expert."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch

import main as training_main
from scripts.evaluate_dual_expert_consensus import collect_logits, metrics, softmax
from scripts.evaluate_motion_expert_consensus import collect_expert_logits
from scripts.evaluate_three_expert_consensus import load_motion_expert
from scripts.train_statistical_motion_expert import load_split
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]


def load_v2_test(seed, output_dir):
    baseline_dir = ROOT / f"experiments/geolife_matched_b0_fixed_60s_seed{seed}"
    four_dir = ROOT / f"experiments/geolife_60s_b2_feature_seed{seed}"
    motion_dir = ROOT / f"experiments/geolife_motion_expert_60s_seed{seed}"
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
    max_len = pipeline.train_data.feature_data.max_seq_len
    _, motion_pipeline, motion_model = load_motion_expert(motion_dir, config)

    b0, four, targets = collect_logits(
        pipeline.model, four_model, pipeline.test_loader,
        pipeline.device, seed, max_len,
    )
    motion, motion_targets = collect_expert_logits(
        motion_model, motion_pipeline.test_loader, motion_pipeline.device,
    )
    if not np.array_equal(targets, motion_targets):
        raise ValueError("V2 test targets are not aligned")
    return (b0, four, motion), targets


def formal_test(seed=10086):
    os.chdir(ROOT)
    output_dir = ROOT / f"experiments/geolife_statistical_motion_seed{seed}"
    validation_path = output_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("accepted_for_formal_test"):
        raise ValueError("Validation gate did not authorize formal testing")
    if validation.get("formal_test_used"):
        raise ValueError("Formal test was already recorded for this calibration")

    logits, targets = load_v2_test(seed, output_dir)
    probabilities = [
        softmax(item, temperature)
        for item, temperature in zip(logits, validation["v2_temperatures"])
    ]
    v2_probability = sum(
        weight * probability
        for weight, probability in zip(validation["v2_weights"], probabilities)
    )

    test_x, statistical_targets = load_split("test", seed, training=False)
    if not np.array_equal(targets, statistical_targets):
        raise ValueError("Statistical and V2 test targets are not aligned")
    model = joblib.load(ROOT / validation["model_path"])
    raw_probability = model.predict_proba(test_x)
    statistical_probability = softmax(
        np.log(raw_probability.clip(1e-9)),
        validation["selected"]["temperature"],
    )
    weight = validation["selected"]["fusion_weight"]
    fusion_probability = (
        (1.0 - weight) * v2_probability
        + weight * statistical_probability
    )

    v2_score = metrics(v2_probability, targets)
    statistical_score = metrics(statistical_probability, targets)
    fusion_score = metrics(fusion_probability, targets)
    result = {
        "protocol": "geolife-low-rate-statistical-motion-formal-test-v1",
        "sampling_interval_seconds": 60,
        "seed": seed,
        "test_samples": int(len(targets)),
        "parameters_frozen_on_validation": True,
        "test": {
            "v2": v2_score,
            "statistical": statistical_score,
            "fusion": fusion_score,
            "delta_vs_v2": {
                name: fusion_score[name] - v2_score[name]
                for name in ("accuracy", "macro_f1")
            },
        },
    }
    (output_dir / "formal_test.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    validation["formal_test_used"] = True
    validation["formal_test_results"] = "formal_test.json"
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    formal_test(args.seed)


if __name__ == "__main__":
    main()
