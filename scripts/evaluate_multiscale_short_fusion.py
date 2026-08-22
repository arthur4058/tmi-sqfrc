"""Validation-selected complementarity check between V2 and short-trajectory net."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_dual_expert_consensus import choose_temperature, metrics, softmax
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_multiscale_short_official import load_split, select_half_noise
from scripts.train_multiscale_short_trajectory import (
    ROOT,
    MultiScaleShortTrajectoryNet,
    make_loader,
    normalize,
)
from scripts.train_statistical_motion_expert import load_v2_validation


def collect_short_probability(split, seed, checkpoint_path, batch_size, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultiScaleShortTrajectoryNet(
        kernels=tuple(checkpoint["kernels"]), summary_pool=True
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    clean, noise, mask, labels, _ = load_split(split)
    values = select_half_noise(clean, noise, seed)
    values = normalize(values, mask, checkpoint["mean"], checkpoint["std"])
    loader = make_loader(
        values, mask, labels, np.ones(len(labels), dtype=np.float32),
        batch_size, False,
    )
    logits = []
    with torch.no_grad():
        for batch_values, batch_mask, _, _ in loader:
            logits.append(model(
                batch_values.to(device), batch_mask.to(device)
            ).cpu().numpy())
    return np.concatenate(logits), labels


def choose_weight(v2_probability, short_probability, labels):
    base = metrics(v2_probability, labels)
    best = None
    for weight in np.linspace(0.0, 0.5, 51):
        score = metrics(
            (1.0 - weight) * v2_probability + weight * short_probability,
            labels,
        )
        delta = [score[key] - base[key] for key in ("accuracy", "macro_f1")]
        candidate = (min(delta), sum(delta), -weight, weight, score)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3], base, best[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    output_dir = ROOT / f"experiments/geolife_multiscale_short_official_seed{args.seed}"
    checkpoint_path = output_dir / "fold0_hybrid_best.pth"

    short_val_logits, val_labels = collect_short_probability(
        "val", args.seed, checkpoint_path, args.batch_size, args.device
    )
    v2_val, v2_val_labels, temperatures, v2_weights = load_v2_validation(args.seed)
    if not np.array_equal(val_labels, v2_val_labels):
        raise ValueError("Validation labels are not aligned")
    short_temperature = choose_temperature(short_val_logits, val_labels)
    short_val = softmax(short_val_logits, short_temperature)
    weight, v2_val_score, fusion_val_score = choose_weight(
        v2_val, short_val, val_labels
    )

    v2_logits, test_labels = load_v2_test(args.seed, output_dir)
    v2_test = sum(
        item_weight * softmax(logit, temperature)
        for item_weight, logit, temperature in zip(
            v2_weights, v2_logits, temperatures
        )
    )
    short_test_logits, short_test_labels = collect_short_probability(
        "test", args.seed, checkpoint_path, args.batch_size, args.device
    )
    if not np.array_equal(test_labels, short_test_labels):
        raise ValueError("Test labels are not aligned")
    short_test = softmax(short_test_logits, short_temperature)
    fusion_test = (1.0 - weight) * v2_test + weight * short_test
    result = {
        "protocol": "geolife-v2-plus-multiscale-short-frozen-v1",
        "seed": args.seed,
        "selection_split": "official user-disjoint validation",
        "short_temperature": short_temperature,
        "short_weight": weight,
        "validation": {
            "v2": v2_val_score,
            "short": metrics(short_val, val_labels),
            "fusion": fusion_val_score,
        },
        "test": {
            "v2": metrics(v2_test, test_labels),
            "short": metrics(short_test, test_labels),
            "fusion": metrics(fusion_test, test_labels),
        },
    }
    for split in ("validation", "test"):
        result[split]["delta_vs_v2"] = {
            key: result[split]["fusion"][key] - result[split]["v2"][key]
            for key in ("accuracy", "macro_f1")
        }
    (output_dir / "v2_frozen_fusion_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
