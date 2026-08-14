"""Evaluate global consensus of baseline and two motion experts."""

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
from scripts.evaluate_motion_expert_consensus import collect_expert_logits
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]


def choose_simplex_mix(probabilities, targets, step=0.02):
    best = None
    count = int(round(1.0 / step))
    for old_units in range(count + 1):
        for motion_units in range(count - old_units + 1):
            weights = np.array([
                (count - old_units - motion_units) / count,
                old_units / count,
                motion_units / count,
            ])
            mixed = sum(w * p for w, p in zip(weights, probabilities))
            score = metrics(mixed, targets)
            candidate = (
                score["accuracy"], score["macro_f1"],
                -float(np.square(weights).sum()), tuple(weights),
            )
            if best is None or candidate > best:
                best = candidate
    return np.array(best[-1])


def load_motion_expert(expert_dir, base_config):
    config = json.loads(
        (expert_dir / "configuration.json").read_text(encoding="utf-8")
    )
    config["test_only"] = "testset"
    config["feature_branch_hyperparams"] = str(
        expert_dir / "feature_branch_classification_from_scratch_model_hyperparams.json"
    )
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    model = model_factory(config, pipeline.train_data)
    checkpoint = torch.load(
        expert_dir / "checkpoints/model_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(pipeline.device)
    return config, pipeline, model


def evaluate(rate):
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
    output_dir = ROOT / f"experiments/geolife_three_expert_{rate}s_seed10086"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        map_location="cpu",
        weights_only=True,
    )
    old_model.load_state_dict(old_checkpoint["state_dict"], strict=True)
    old_model.to(pipeline.device)
    old_max_len = pipeline.train_data.feature_data.max_seq_len

    motion_config, motion_pipeline, motion_model = load_motion_expert(
        motion_dir, b0_config
    )
    val_b0_logits, val_old_logits, val_targets = collect_logits(
        pipeline.model, old_model, pipeline.val_loader,
        pipeline.device, b0_config["seed"], old_max_len,
    )
    test_b0_logits, test_old_logits, test_targets = collect_logits(
        pipeline.model, old_model, pipeline.test_loader,
        pipeline.device, b0_config["seed"], old_max_len,
    )
    val_motion_logits, val_motion_targets = collect_expert_logits(
        motion_model, motion_pipeline.val_loader, motion_pipeline.device
    )
    test_motion_logits, test_motion_targets = collect_expert_logits(
        motion_model, motion_pipeline.test_loader, motion_pipeline.device
    )
    if not np.array_equal(val_targets, val_motion_targets):
        raise ValueError("Validation targets are not aligned")
    if not np.array_equal(test_targets, test_motion_targets):
        raise ValueError("Test targets are not aligned")

    val_logits = [val_b0_logits, val_old_logits, val_motion_logits]
    test_logits = [test_b0_logits, test_old_logits, test_motion_logits]
    temperatures = [
        choose_temperature(logits, val_targets) for logits in val_logits
    ]
    val_prob = [
        softmax(logits, temperature)
        for logits, temperature in zip(val_logits, temperatures)
    ]
    test_prob = [
        softmax(logits, temperature)
        for logits, temperature in zip(test_logits, temperatures)
    ]
    weights = choose_simplex_mix(val_prob, val_targets)

    def summarize(probabilities, targets):
        mixed = sum(w * p for w, p in zip(weights, probabilities))
        return {
            "baseline": metrics(probabilities[0], targets),
            "four_feature_expert": metrics(probabilities[1], targets),
            "motion_expert": metrics(probabilities[2], targets),
            "consensus": metrics(mixed, targets),
        }

    result = {
        "protocol": "geolife-low-rate-three-expert-v2",
        "sampling_interval_seconds": rate,
        "selection_split": "user-disjoint-validation",
        "test_touched_during_selection": False,
        "temperatures": temperatures,
        "weights": {
            "baseline": float(weights[0]),
            "four_feature_expert": float(weights[1]),
            "motion_expert": float(weights[2]),
        },
        "validation": summarize(val_prob, val_targets),
        "test": summarize(test_prob, test_targets),
    }
    result["test_delta"] = {
        name: result["test"]["consensus"][name]
        - result["test"]["baseline"][name]
        for name in ("accuracy", "macro_f1")
    }
    output = output_dir / "three_expert_consensus_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, choices=(30, 60), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    evaluate(args.rate)


if __name__ == "__main__":
    main()
