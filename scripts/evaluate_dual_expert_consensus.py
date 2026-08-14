"""Evaluate validation-calibrated dual-expert consensus at one sampling rate."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import main as training_main
from tmi.datasets import dataset
from tmi.models.models import model_factory


ROOT = Path(__file__).resolve().parents[1]


def metrics(probabilities, targets):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def softmax(logits, temperature=1.0):
    logits = logits / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def choose_temperature(logits, targets):
    best = None
    for temperature in np.linspace(0.5, 3.0, 101):
        probabilities = softmax(logits, float(temperature))
        nll = -np.log(
            probabilities[np.arange(len(targets)), targets].clip(1e-9)
        ).mean()
        candidate = (float(nll), float(temperature))
        if best is None or candidate < best:
            best = candidate
    return best[1]


def choose_uniform_mix(prob_b0, prob_b2, targets):
    best = None
    for alpha in np.linspace(0.0, 1.0, 101):
        score = metrics((1.0 - alpha) * prob_b0 + alpha * prob_b2, targets)
        candidate = (score["accuracy"], score["macro_f1"], float(alpha))
        if best is None or candidate > best:
            best = candidate
    return best[2]


def paths(rate):
    baseline = ROOT / (
        f"experiments/geolife_matched_b0_fixed_{rate}s_seed10086/"
        "matched_test/configuration.json"
    )
    expert_dir = (
        ROOT / "experiments/geolife_60s_b2_feature_seed10086"
        if rate == 60
        else ROOT / f"experiments/geolife_feature_expert_{rate}s_seed10086"
    )
    return baseline, expert_dir


def pad_for_expert(features, masks, max_len):
    """Pad or clip a dynamic dual-branch batch for a fixed-width expert."""
    if features.shape[1] < max_len:
        pad_len = max_len - features.shape[1]
        features = torch.nn.functional.pad(features, (0, 0, 0, pad_len))
        masks = torch.nn.functional.pad(masks, (0, pad_len), value=False)
    elif features.shape[1] > max_len:
        features = features[:, :max_len]
        masks = masks[:, :max_len]
    return features, masks


def collect_logits(model_b0, model_b2, loader, device, seed, b2_max_len):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_b0.eval()
    model_b2.eval()
    b0_logits, b2_logits, targets = [], [], []
    with torch.inference_mode():
        for x1, x2, mask1, mask2, target, _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)
            b2_x2, b2_mask2 = pad_for_expert(x2, mask2, b2_max_len)
            b0_logits.append(model_b0(x1, mask1, x2, mask2).cpu())
            b2_logits.append(
                model_b2(b2_x2, b2_mask2, feature_masks=None).cpu()
            )
            targets.append(target.reshape(-1).cpu())
    return (
        torch.cat(b0_logits).numpy(),
        torch.cat(b2_logits).numpy(),
        torch.cat(targets).numpy(),
    )


def evaluate(rate):
    baseline_path, expert_dir = paths(rate)
    b0_config = json.loads(baseline_path.read_text(encoding="utf-8"))
    b2_config = json.loads(
        (expert_dir / "configuration.json").read_text(encoding="utf-8")
    )
    output_dir = ROOT / f"experiments/geolife_consensus_{rate}s_seed10086"
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

    b2_config["test_only"] = None
    b2 = model_factory(b2_config, pipeline.train_data.feature_data)
    checkpoint = torch.load(
        expert_dir / "checkpoints/model_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    b2.load_state_dict(checkpoint["state_dict"], strict=True)
    b2.to(pipeline.device)
    b2_max_len = pipeline.train_data.feature_data.max_seq_len

    validation = collect_logits(
        pipeline.model, b2, pipeline.val_loader, pipeline.device,
        b0_config["seed"], b2_max_len,
    )
    testing = collect_logits(
        pipeline.model, b2, pipeline.test_loader, pipeline.device,
        b0_config["seed"], b2_max_len,
    )
    val_b0_logits, val_b2_logits, val_targets = validation
    test_b0_logits, test_b2_logits, test_targets = testing
    t_b0 = choose_temperature(val_b0_logits, val_targets)
    t_b2 = choose_temperature(val_b2_logits, val_targets)
    val_b0 = softmax(val_b0_logits, t_b0)
    val_b2 = softmax(val_b2_logits, t_b2)
    test_b0 = softmax(test_b0_logits, t_b0)
    test_b2 = softmax(test_b2_logits, t_b2)
    alpha = choose_uniform_mix(val_b0, val_b2, val_targets)

    def summarize(prob_b0, prob_b2, targets):
        mixed = (1.0 - alpha) * prob_b0 + alpha * prob_b2
        return {
            "baseline": metrics(prob_b0, targets),
            "feature_expert": metrics(prob_b2, targets),
            "consensus": metrics(mixed, targets),
        }

    result = {
        "protocol": "geolife-five-rate-dual-expert-consensus-v1",
        "sampling_interval_seconds": rate,
        "selection_split": "user-disjoint-validation",
        "test_touched_during_selection": False,
        "temperature": {"baseline": t_b0, "feature_expert": t_b2},
        "alpha_feature_expert": alpha,
        "validation": summarize(val_b0, val_b2, val_targets),
        "test": summarize(test_b0, test_b2, test_targets),
    }
    result["test_delta"] = {
        name: result["test"]["consensus"][name]
        - result["test"]["baseline"][name]
        for name in ("accuracy", "macro_f1")
    }
    output = output_dir / "consensus_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, choices=(5, 10, 20, 30, 60), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    evaluate(args.rate)


if __name__ == "__main__":
    main()
