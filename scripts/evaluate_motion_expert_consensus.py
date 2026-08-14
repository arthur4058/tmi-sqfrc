"""Evaluate a validation-calibrated low-rate motion-feature expert."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import main as training_main
from scripts.evaluate_dual_expert_consensus import (
    choose_temperature,
    choose_uniform_mix,
    metrics,
    softmax,
)
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]


def collect_baseline_logits(model, loader, device):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for x1, x2, mask1, mask2, target, _ in loader:
            logits.append(model(
                x1.to(device), mask1.to(device),
                x2.to(device), mask2.to(device),
            ).cpu())
            targets.append(target.reshape(-1).cpu())
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def collect_expert_logits(model, loader, device):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for x, target, padding_masks, _ in loader:
            logits.append(model(
                x.to(device), padding_masks.to(device), feature_masks=None,
            ).cpu())
            targets.append(target.reshape(-1).cpu())
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def evaluate(rate, expert_dir, output_dir):
    baseline_path = ROOT / (
        f"experiments/geolife_matched_b0_fixed_{rate}s_seed10086/"
        "matched_test/configuration.json"
    )
    expert_dir = ROOT / expert_dir
    output_dir = ROOT / output_dir
    b0_config = json.loads(baseline_path.read_text(encoding="utf-8"))
    expert_config = json.loads(
        (expert_dir / "configuration.json").read_text(encoding="utf-8")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    b0_config.update({
        "output_dir": str(output_dir),
        "records_file": str(output_dir / "records.xlsx"),
    })
    dataset.config = b0_config

    pipeline = training_main.TrainingPipeline(b0_config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()

    expert_config["test_only"] = "testset"
    expert_config["feature_branch_hyperparams"] = str(
        expert_dir / "feature_branch_classification_from_scratch_model_hyperparams.json"
    )
    dataset.config = expert_config
    expert_pipeline = training_main.TrainingPipeline(expert_config)
    expert_pipeline.setup_data()
    expert_pipeline.prepare_data_loaders()
    expert = model_factory(expert_config, expert_pipeline.train_data)
    checkpoint = torch.load(
        expert_dir / "checkpoints/model_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    expert.load_state_dict(checkpoint["state_dict"], strict=True)
    expert.to(expert_pipeline.device)

    val_b0_logits, val_targets = collect_baseline_logits(
        pipeline.model, pipeline.val_loader, pipeline.device
    )
    test_b0_logits, test_targets = collect_baseline_logits(
        pipeline.model, pipeline.test_loader, pipeline.device
    )
    val_expert_logits, val_expert_targets = collect_expert_logits(
        expert, expert_pipeline.val_loader, expert_pipeline.device
    )
    test_expert_logits, test_expert_targets = collect_expert_logits(
        expert, expert_pipeline.test_loader, expert_pipeline.device
    )
    if not np.array_equal(val_targets, val_expert_targets):
        raise ValueError("Validation targets are not aligned between models")
    if not np.array_equal(test_targets, test_expert_targets):
        raise ValueError("Test targets are not aligned between models")

    t_b0 = choose_temperature(val_b0_logits, val_targets)
    t_expert = choose_temperature(val_expert_logits, val_targets)
    val_b0 = softmax(val_b0_logits, t_b0)
    val_expert = softmax(val_expert_logits, t_expert)
    test_b0 = softmax(test_b0_logits, t_b0)
    test_expert = softmax(test_expert_logits, t_expert)
    alpha = choose_uniform_mix(val_b0, val_expert, val_targets)

    def summarize(prob_b0, prob_expert, targets):
        mixed = (1.0 - alpha) * prob_b0 + alpha * prob_expert
        return {
            "baseline": metrics(prob_b0, targets),
            "motion_expert": metrics(prob_expert, targets),
            "consensus": metrics(mixed, targets),
        }

    result = {
        "protocol": "geolife-low-rate-motion-expert-v2",
        "sampling_interval_seconds": rate,
        "selection_split": "user-disjoint-validation",
        "test_touched_during_selection": False,
        "motion_features": expert_config["motion_features"],
        "temperature": {"baseline": t_b0, "motion_expert": t_expert},
        "alpha_motion_expert": alpha,
        "validation": summarize(val_b0, val_expert, val_targets),
        "test": summarize(test_b0, test_expert, test_targets),
    }
    result["test_delta"] = {
        name: result["test"]["consensus"][name]
        - result["test"]["baseline"][name]
        for name in ("accuracy", "macro_f1")
    }
    output = output_dir / "motion_expert_consensus_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, choices=(30, 60), required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    evaluate(args.rate, args.expert_dir, args.output_dir)


if __name__ == "__main__":
    main()
