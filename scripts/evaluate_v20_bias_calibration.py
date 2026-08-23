"""Validation-only class-bias calibration for the V20 60-second model."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import main as training_main
from tmi.datasets import dataset


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/geolife_v20_unweighted_calibration_60s_seed10086"


def score(logits, targets, bias):
    predictions = (logits + bias).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def collect_logits(model, loader, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for x1, x2, mask1, mask2, target, _ in loader:
            values = model(
                x1.to(device), mask1.to(device),
                x2.to(device), mask2.to(device),
            )
            logits.append(values.cpu())
            targets.append(target.reshape(-1).cpu())
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def choose_bias(logits, targets):
    """Coordinate search with one reference class and a strict small bound."""
    bias = np.zeros(logits.shape[1], dtype=np.float64)
    baseline = score(logits, targets, bias)
    best_key = (baseline["accuracy"], baseline["macro_f1"], 0.0)
    grid = np.arange(-0.15, 0.1501, 0.01)
    for _ in range(4):
        changed = False
        for class_index in range(1, logits.shape[1]):
            local_bias = bias.copy()
            local_key = best_key
            for value in grid:
                candidate_bias = bias.copy()
                candidate_bias[class_index] = value
                candidate = score(logits, targets, candidate_bias)
                key = (
                    candidate["accuracy"],
                    candidate["macro_f1"],
                    -float(np.square(candidate_bias).sum()),
                )
                if key > local_key:
                    local_key = key
                    local_bias = candidate_bias
            if not np.array_equal(local_bias, bias):
                changed = True
                bias = local_bias
                best_key = local_key
        if not changed:
            break
    return bias


def main():
    os.chdir(ROOT)
    config = json.loads(
        (EXPERIMENT / "configuration.json").read_text(encoding="utf-8")
    )
    output_dir = ROOT / "experiments/geolife_v20_calibrated_60s_seed10086"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.update({
        "task": "dual_branch_classification",
        "test_only": "testset",
        "load_model": str(EXPERIMENT / "checkpoints/model_best.pth"),
        "initialize_base_model": None,
        "output_dir": str(output_dir),
        "records_file": str(output_dir / "records.xlsx"),
    })
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()

    val_logits, val_targets = collect_logits(
        pipeline.model, pipeline.val_loader, pipeline.device, config["seed"]
    )
    bias = choose_bias(val_logits, val_targets)
    validation = {
        "uncalibrated": score(val_logits, val_targets, np.zeros(5)),
        "calibrated": score(val_logits, val_targets, bias),
    }

    # The test split is read only after all calibration parameters are frozen.
    test_logits, test_targets = collect_logits(
        pipeline.model, pipeline.test_loader, pipeline.device, config["seed"]
    )
    testing = {
        "uncalibrated": score(test_logits, test_targets, np.zeros(5)),
        "calibrated": score(test_logits, test_targets, bias),
    }
    result = {
        "protocol": "geolife-v20-validation-bias-calibration",
        "sampling_interval_seconds": 60,
        "seed": config["seed"],
        "selection_split": "user-disjoint-validation",
        "test_touched_during_selection": False,
        "class_bias": bias.tolist(),
        "validation": validation,
        "test": testing,
        "test_delta": {
            key: testing["calibrated"][key] - testing["uncalibrated"][key]
            for key in ("accuracy", "macro_f1")
        },
    }
    target = output_dir / "calibrated_results.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
