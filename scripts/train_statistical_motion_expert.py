"""Train validation-selected low-rate statistical experts without touching test."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

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
DATA_ROOT = ROOT / "data/geolife_five_rate_fixed_60s_features"
# Keep only relative motion channels. Absolute timestamp/hour and coordinates
# would leak a user's collection period/region across user-disjoint splits.
FEATURE_INDICES = (2, 3, 4, 5, 6, 7, 8)


def _finite(values):
    return np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def _channel_summary(values):
    values = _finite(values).reshape(-1)
    if values.size == 0:
        return np.zeros(13, dtype=np.float32)
    differences = np.diff(values)
    mean_abs_difference = (
        float(np.abs(differences).mean()) if differences.size else 0.0
    )
    max_abs_difference = (
        float(np.abs(differences).max()) if differences.size else 0.0
    )
    if values.size > 1:
        position = np.linspace(-1.0, 1.0, values.size)
        slope = float(np.polyfit(position, values, 1)[0])
    else:
        slope = 0.0
    return np.asarray([
        values.mean(), values.std(), values.min(), values.max(),
        np.median(values), np.quantile(values, 0.25),
        np.quantile(values, 0.75), values[0], values[-1],
        values[-1] - values[0], mean_abs_difference,
        max_abs_difference, slope,
    ], dtype=np.float32)


def _trajectory_summary(sample):
    latitude = _finite(sample[0]).reshape(-1)
    longitude = _finite(sample[1]).reshape(-1)
    length = min(len(latitude), len(longitude))
    if length == 0:
        return np.zeros(10, dtype=np.float32)
    latitude, longitude = latitude[:length], longitude[:length]
    if length == 1:
        return np.asarray([
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

    lat_mid = np.deg2rad(0.5 * (latitude[1:] + latitude[:-1]))
    dy = np.diff(latitude) * 111_320.0
    dx = np.diff(longitude) * 111_320.0 * np.cos(lat_mid)
    step = np.sqrt(dx * dx + dy * dy)
    path = float(step.sum())
    net = float(math.hypot(dx.sum(), dy.sum()))
    heading = np.unwrap(np.arctan2(dy, dx))
    turn = np.abs(np.diff(heading))
    return np.asarray([
        float(length), path, net, net / max(path, 1e-6),
        step.mean(), step.std(), step.max(),
        float(np.median(step)),
        float(turn.mean()) if turn.size else 0.0,
        float(turn.max()) if turn.size else 0.0,
    ], dtype=np.float32)


def summarize_arrays(features, trajectories):
    output = np.empty(
        (len(features), len(FEATURE_INDICES) * 13 + 10),
        dtype=np.float32,
    )
    for row in range(len(features)):
        summaries = [
            _channel_summary(features[row, index])
            for index in FEATURE_INDICES
        ]
        summaries.append(_trajectory_summary(trajectories[row]))
        output[row] = np.concatenate(summaries)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def load_split(split, seed, training=False):
    base = DATA_ROOT / split
    clean_features = np.load(
        base / "clean_multi_feature_segs.npy", allow_pickle=True)
    noise_features = np.load(
        base / "noise_multi_feature_segs.npy", allow_pickle=True)
    clean_trajectories = np.load(
        base / "clean_trj_segs.npy", allow_pickle=True)
    noise_trajectories = np.load(
        base / "noise_trj_segs.npy", allow_pickle=True)
    labels = np.load(
        base / "clean_multi_feature_seg_labels.npy", allow_pickle=True
    ).astype(np.int64)

    clean = summarize_arrays(clean_features, clean_trajectories)
    noise = summarize_arrays(noise_features, noise_trajectories)
    if training:
        return np.concatenate([clean, noise]), np.concatenate([labels, labels])

    generator = random.Random(int(seed))
    use_noise = np.fromiter(
        (generator.random() < 0.5 for _ in range(len(labels))),
        dtype=bool, count=len(labels),
    )
    selected = clean.copy()
    selected[use_noise] = noise[use_noise]
    return selected, labels


def load_v2_validation(seed):
    baseline_dir = ROOT / f"experiments/geolife_matched_b0_fixed_60s_seed{seed}"
    four_dir = ROOT / f"experiments/geolife_60s_b2_feature_seed{seed}"
    motion_dir = ROOT / f"experiments/geolife_motion_expert_60s_seed{seed}"
    output_dir = ROOT / f"experiments/geolife_statistical_motion_seed{seed}"
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
    max_len = pipeline.train_data.feature_data.max_seq_len
    _, motion_pipeline, motion_model = load_motion_expert(motion_dir, config)

    b0, four, targets = collect_logits(
        pipeline.model, four_model, pipeline.val_loader,
        pipeline.device, seed, max_len,
    )
    motion, motion_targets = collect_expert_logits(
        motion_model, motion_pipeline.val_loader, motion_pipeline.device,
    )
    if not np.array_equal(targets, motion_targets):
        raise ValueError("V2 validation targets are not aligned")
    temperatures = [choose_temperature(item, targets) for item in (b0, four, motion)]
    probabilities = [
        softmax(item, temperature)
        for item, temperature in zip((b0, four, motion), temperatures)
    ]
    weights = choose_simplex_mix(probabilities, targets)
    probability = sum(w * p for w, p in zip(weights, probabilities))
    return probability, targets, temperatures, weights


def choose_mix(v2_probability, expert_probability, targets):
    base = metrics(v2_probability, targets)
    best = None
    for weight in np.linspace(0.0, 0.7, 71):
        score = metrics(
            (1.0 - weight) * v2_probability + weight * expert_probability,
            targets,
        )
        delta_accuracy = score["accuracy"] - base["accuracy"]
        delta_f1 = score["macro_f1"] - base["macro_f1"]
        candidate = (
            min(delta_accuracy, delta_f1),
            delta_accuracy + delta_f1,
            -float(weight), float(weight), score,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3], base, best[4]


def candidates(seed):
    return {
        "extra_trees_deep": ExtraTreesClassifier(
            n_estimators=350, max_features=0.7, min_samples_leaf=2,
            max_depth=24, class_weight="balanced", n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees_robust": ExtraTreesClassifier(
            n_estimators=350, max_features="sqrt", min_samples_leaf=5,
            max_depth=20, class_weight="balanced", n_jobs=-1,
            random_state=seed + 1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=220, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=0.5,
            class_weight="balanced", random_state=seed,
        ),
    }


def train(seed=10086):
    os.chdir(ROOT)
    output_dir = ROOT / f"experiments/geolife_statistical_motion_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split("train", seed, training=True)
    val_x, val_y = load_split("val", seed, training=False)
    v2_probability, v2_targets, v2_temperatures, v2_weights = (
        load_v2_validation(seed)
    )
    if not np.array_equal(val_y, v2_targets):
        raise ValueError("Statistical and V2 validation targets are not aligned")

    trials = []
    best = None
    for name, model in candidates(seed).items():
        model.fit(train_x, train_y)
        raw_probability = model.predict_proba(val_x)
        temperature = choose_temperature(
            np.log(raw_probability.clip(1e-9)), val_y
        )
        probability = softmax(
            np.log(raw_probability.clip(1e-9)), temperature
        )
        weight, v2_score, fusion_score = choose_mix(
            v2_probability, probability, val_y
        )
        delta = {
            key: fusion_score[key] - v2_score[key]
            for key in ("accuracy", "macro_f1")
        }
        trial = {
            "name": name,
            "temperature": temperature,
            "fusion_weight": weight,
            "standalone": metrics(probability, val_y),
            "fusion": fusion_score,
            "delta_vs_v2": delta,
        }
        trials.append(trial)
        rank = (min(delta.values()), sum(delta.values()))
        if best is None or rank > best[0]:
            best = (rank, name, model, trial)

    _, name, model, selected = best
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    result = {
        "protocol": "geolife-low-rate-statistical-motion-validation-v2",
        "location_and_timestamp_invariant": True,
        "seed": seed,
        "sampling_interval_seconds": 60,
        "features": int(train_x.shape[1]),
        "training_rows": int(train_x.shape[0]),
        "formal_test_used": False,
        "v2_temperatures": v2_temperatures,
        "v2_weights": [float(value) for value in v2_weights],
        "v2_validation": metrics(v2_probability, val_y),
        "trials": trials,
        "selected": selected,
        "selected_model": name,
        "model_path": str(model_path.relative_to(ROOT)),
        "accepted_for_formal_test": (
            selected["delta_vs_v2"]["accuracy"] >= 0.01
            and selected["delta_vs_v2"]["macro_f1"] >= 0.01
        ),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    train(args.seed)


if __name__ == "__main__":
    main()
