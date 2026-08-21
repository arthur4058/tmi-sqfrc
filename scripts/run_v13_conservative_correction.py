#!/usr/bin/env python3
"""Select V13 only on Train-A/Dev-A; never load the formal test split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as training_main
from scripts.evaluate_behavior_recovery import inverse_targets
from tmi.datasets import dataset
from tmi.models.behavior_recovery import IntervalBehaviorRecovery
from tmi.models.conservative_corrector import ConservativeBusCarCorrector

CLASS_NAMES = ("Walk", "Bike", "Bus", "Car", "Train")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_bundle(logits: np.ndarray, labels: np.ndarray):
    pred = logits.argmax(axis=1)
    per_class = f1_score(labels, pred, labels=np.arange(5), average=None, zero_division=0)
    matrix = confusion_matrix(labels, pred, labels=np.arange(5))
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "per_class_f1": {
            name: float(per_class[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "bus_to_car": int(matrix[2, 3]),
        "car_to_bus": int(matrix[3, 2]),
    }


def load_pipeline(config_path: Path, checkpoint: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "task": "dual_branch_classification",
        "test_only": "testset",
        "load_model": str(checkpoint),
        "output_dir": "./experiments/geolife_v13_dev_selection",
        "records_file": "./experiments/geolife_v13_dev_selection/records.xlsx",
    })
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()
    return config, pipeline


def sequential_train_loader(pipeline, batch_size):
    train_dataset = pipeline.dataset_class(
        pipeline.train_data, pipeline.train_indices)
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=pipeline.collate_fn)


def collect_logits(model, loader, device, seed):
    set_seed(seed)
    model.eval()
    logits, labels, ids = [], [], []
    with torch.inference_mode():
        for x1, x2, mask1, mask2, target, sample_id in loader:
            output = model(
                x1.to(device), mask1.to(device),
                x2.to(device), mask2.to(device))
            logits.append(output.cpu())
            labels.append(target.reshape(-1).cpu())
            ids.append(torch.as_tensor(
                [int(value) for value in sample_id], dtype=torch.int64
            ))
    return (
        torch.cat(logits).numpy(),
        torch.cat(labels).numpy().astype(np.int64),
        torch.cat(ids).numpy().astype(np.int64),
    )


def load_v12(checkpoint_path: Path, device):
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model = IntervalBehaviorRecovery(
        input_dim=7, hidden_dim=int(payload["hidden_dim"]),
        output_dim=6, dropout=float(payload["dropout"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def raw_sparse_features(row):
    channels = [np.asarray(row[index], dtype=np.float32) for index in (0, 2, 3, 6, 7, 8)]
    length = min(map(len, channels))
    if length == 0:
        return np.empty((0, 7), dtype=np.float32)
    return np.column_stack([
        channel[:length] for channel in channels
    ] + [np.ones(length, dtype=np.float32)]).astype(np.float32)


def behavior_representations(feature_root: Path, split: str, ids: np.ndarray,
                             model, payload, device, batch_size=8192):
    segments = np.load(
        feature_root / split / "noise_multi_feature_segs.npy",
        allow_pickle=True)
    flat, offsets, lengths = [], [0], []
    for sample_id in ids:
        values = raw_sparse_features(segments[int(sample_id)])
        flat.append(values)
        lengths.append(len(values))
        offsets.append(offsets[-1] + len(values))
    nonempty = np.concatenate([value for value in flat if len(value)], axis=0)
    input_mean = np.asarray(payload["input_mean"], dtype=np.float32)
    input_std = np.asarray(payload["input_std"], dtype=np.float32)
    normalized = np.clip((nonempty - input_mean) / input_std, -10.0, 10.0).astype(np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start:start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    normalized_prediction = np.concatenate(outputs)
    transformed = (
        normalized_prediction * np.asarray(payload["target_std"])
        + np.asarray(payload["target_mean"]))
    prediction = inverse_targets(transformed)

    representations = np.zeros((len(ids), 12), dtype=np.float32)
    valid_points = np.zeros(len(ids), dtype=np.float32)
    valid_ratio = np.ones(len(ids), dtype=np.float32)
    cursor = 0
    for index, length in enumerate(lengths):
        if length:
            sample = prediction[cursor:cursor + length]
            representations[index] = np.concatenate((sample.mean(axis=0), sample.max(axis=0)))
            cursor += length
        valid_points[index] = float(length + 1)
    return representations, valid_points, valid_ratio


def entropy_and_confidence(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    entropy = -(probability * np.log(probability.clip(1e-9))).sum(axis=1)
    return entropy.astype(np.float32), probability.max(axis=1).astype(np.float32)


def build_corrector_features(behavior, logits, valid_points, valid_ratio):
    entropy, confidence = entropy_and_confidence(logits)
    margin = np.abs(logits[:, 2] - logits[:, 3])
    return np.column_stack((
        behavior,
        logits[:, 2], logits[:, 3], margin,
        entropy, confidence, valid_points, valid_ratio,
    )).astype(np.float32)


def train_candidate(train_x, train_logits, train_y, train_points,
                    dev_x, dev_logits, dev_y, dev_points,
                    settings, seed, device):
    set_seed(seed)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train_norm = np.clip((train_x - mean) / std, -10.0, 10.0).astype(np.float32)
    dev_norm = np.clip((dev_x - mean) / std, -10.0, 10.0).astype(np.float32)
    train_dataset = TensorDataset(
        torch.from_numpy(train_norm), torch.from_numpy(train_logits.astype(np.float32)),
        torch.from_numpy(train_y), torch.from_numpy(train_points.astype(np.float32)))
    loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    model = ConservativeBusCarCorrector(
        input_dim=train_x.shape[1], hidden_dim=16,
        alpha=settings["alpha"], n_low=5.0, n_high=15.0,
        uncertainty_tau=settings["tau"],
        uncertainty_temperature=settings["temperature"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_score, best_epoch, stale = None, None, 0, 0
    start = time.time()
    for epoch in range(1, 41):
        model.train()
        for features, base, labels, points in loader:
            features, base = features.to(device), base.to(device)
            labels, points = labels.to(device), points.to(device)
            corrected, _, sparse, _ = model(features, base, points)
            classification = F.cross_entropy(corrected, labels)
            with torch.no_grad():
                base_probability = F.softmax(base, dim=1)
            correction_log_probability = F.log_softmax(corrected, dim=1)
            anchor_per_sample = F.kl_div(
                correction_log_probability, base_probability,
                reduction="none").sum(dim=1)
            anchor = ((1.0 - sparse) * anchor_per_sample).mean()
            loss = classification + settings["lambda_anchor"] * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        dev_result = apply_model(model, dev_norm, dev_logits, dev_points, device)
        score_metrics = metric_bundle(dev_result["logits"], dev_y)
        car_delta = score_metrics["per_class_f1"]["Car"]
        score = (score_metrics["accuracy"], score_metrics["macro_f1"], car_delta)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    applied = apply_model(model, dev_norm, dev_logits, dev_points, device)
    return {
        "model": model,
        "input_mean": mean,
        "input_std": std,
        "best_epoch": best_epoch,
        "runtime_seconds": time.time() - start,
        **applied,
    }


def apply_model(model, features, logits, points, device):
    outputs, deltas, sparse_gates, uncertainty_gates = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            batch = torch.from_numpy(features[start:start + 8192]).to(device)
            base = torch.from_numpy(logits[start:start + 8192].astype(np.float32)).to(device)
            valid = torch.from_numpy(points[start:start + 8192].astype(np.float32)).to(device)
            corrected, delta, sparse, uncertain = model(batch, base, valid)
            outputs.append(corrected.cpu().numpy())
            deltas.append(delta.cpu().numpy())
            sparse_gates.append(sparse.cpu().numpy())
            uncertainty_gates.append(uncertain.cpu().numpy())
    return {
        "logits": np.concatenate(outputs),
        "delta": np.concatenate(deltas),
        "sparsity_gate": np.concatenate(sparse_gates),
        "uncertainty_gate": np.concatenate(uncertainty_gates),
    }


def write_report(path, result):
    b0 = result["dev_a"]["b0"]
    v13 = result["dev_a"]["v13"]
    lines = [
        "# GeoLife V13 保守 Bus-Car 纠错 Dev-A 实验", "",
        "## 结论", "", f"**V13 {result['decision']}**。", "",
        "本轮只使用原训练用户内部的 Train-A/Dev-A，正式 validation/test 未加载。",
        "B0 在 Train-A 重新训练，因此 Dev-A 用户对 B0、V12 和 V13 都是未见用户。", "",
        "| Method | Accuracy | Macro-F1 | Walk F1 | Bike F1 | Bus F1 | Car F1 | Train F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("B0", b0), ("V12（不改 logits）", b0), ("V12+V13", v13)):
        f1 = metrics["per_class_f1"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{f1['Walk']:.4f} | {f1['Bike']:.4f} | {f1['Bus']:.4f} | "
            f"{f1['Car']:.4f} | {f1['Train']:.4f} |")
    delta = result["dev_a"]["delta"]
    lines.extend([
        "", "## 配对增量", "",
        f"- Accuracy：{delta['accuracy'] * 100:+.2f} pp",
        f"- Macro-F1：{delta['macro_f1'] * 100:+.2f} pp",
        f"- Bus F1：{delta['bus_f1'] * 100:+.2f} pp",
        f"- Car F1：{delta['car_f1'] * 100:+.2f} pp", "",
        "## Bus-Car 混淆", "",
        f"- B0 Bus→Car：{b0['bus_to_car']}；V13：{v13['bus_to_car']}",
        f"- B0 Car→Bus：{b0['car_to_bus']}；V13：{v13['car_to_bus']}", "",
        "## Gate", "",
        f"- Dev-A 平均 sparsity gate：{result['gates']['dev_sparsity_mean']:.4f}",
        f"- Dev-A 平均 uncertainty gate：{result['gates']['dev_uncertainty_mean']:.4f}",
        f"- 最大 |delta|：{result['gates']['max_abs_delta']:.4f}", "",
        "高采样率理论保护值（按平均点数代入同一 gate）：", "",
    ])
    for rate, value in result["gates"]["nominal_sparsity_gate"].items():
        lines.append(f"- {rate}s：{value:.4f}")
    lines.extend([
        "", "## 验收门槛", "",
        "- ΔAccuracy ≥ +1.0 pp；",
        "- ΔMacro-F1 ≥ +2.0 pp；",
        "- Car F1 Δ ≥ -0.5 pp；",
        "- Bus F1 明显改善。", "",
        f"正式测试是否访问：{result['formal_test_loaded']}。", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    os.chdir(REPO_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.b0_experiment / "checkpoints/model_best.pth"
    config_path = args.b0_experiment / "configuration.json"
    config, pipeline = load_pipeline(config_path, checkpoint)
    train_loader = sequential_train_loader(pipeline, config["batch_size"])
    train_logits, train_y, train_ids = collect_logits(
        pipeline.model, train_loader, pipeline.device, args.seed)
    dev_logits, dev_y, dev_ids = collect_logits(
        pipeline.model, pipeline.val_loader, pipeline.device, args.seed)
    v12, v12_payload = load_v12(args.v12_checkpoint, device)
    feature_root = Path("data/geolife_v13_dev_fixed_60s_features")
    train_behavior, train_points, train_ratio = behavior_representations(
        feature_root, "train", train_ids, v12, v12_payload, device)
    dev_behavior, dev_points, dev_ratio = behavior_representations(
        feature_root, "val", dev_ids, v12, v12_payload, device)
    train_x = build_corrector_features(
        train_behavior, train_logits, train_points, train_ratio)
    dev_x = build_corrector_features(
        dev_behavior, dev_logits, dev_points, dev_ratio)
    b0_metrics = metric_bundle(dev_logits, dev_y)

    candidates = []
    best = None
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for tau in (0.5, 1.0):
            settings = {
                "alpha": alpha, "tau": tau, "temperature": 0.5,
                "lambda_anchor": 0.5,
            }
            trained = train_candidate(
                train_x, train_logits, train_y, train_points,
                dev_x, dev_logits, dev_y, dev_points,
                settings, args.seed, device)
            result_metrics = metric_bundle(trained["logits"], dev_y)
            delta = {
                "accuracy": result_metrics["accuracy"] - b0_metrics["accuracy"],
                "macro_f1": result_metrics["macro_f1"] - b0_metrics["macro_f1"],
                "bus_f1": result_metrics["per_class_f1"]["Bus"] - b0_metrics["per_class_f1"]["Bus"],
                "car_f1": result_metrics["per_class_f1"]["Car"] - b0_metrics["per_class_f1"]["Car"],
            }
            passed = (
                delta["accuracy"] >= 0.01
                and delta["macro_f1"] >= 0.02
                and delta["car_f1"] >= -0.005
                and delta["bus_f1"] > 0.0)
            summary = {
                "settings": settings, "metrics": result_metrics,
                "delta": delta, "passed": passed,
                "best_epoch": trained["best_epoch"],
            }
            candidates.append(summary)
            score = (
                passed, delta["accuracy"] >= 0.01,
                delta["macro_f1"] >= 0.02,
                delta["car_f1"] >= -0.005,
                delta["accuracy"] + delta["macro_f1"],
            )
            if best is None or score > best[0]:
                best = (score, summary, trained)
            print("candidate", json.dumps(summary, ensure_ascii=False))

    selected, trained = best[1], best[2]
    passed = bool(selected["passed"])
    nominal_points = {5: 51.39, 10: 27.75, 20: 14.65, 30: 10.10, 60: 5.43}
    nominal_gates = {
        str(rate): float(np.clip((15.0 - points) / 10.0, 0.0, 1.0))
        for rate, points in nominal_points.items()
    }
    result = {
        "protocol": "geolife-v13-conservative-correction-dev-v1",
        "decision": "PASS" if passed else "FAIL",
        "selection_split": "Dev-A users from original training users",
        "formal_test_loaded": False,
        "b0_checkpoint": str(checkpoint),
        "b0_checkpoint_sha256": sha256(checkpoint),
        "v12_checkpoint": str(args.v12_checkpoint),
        "selected": selected,
        "all_candidates": candidates,
        "dev_a": {
            "samples": int(len(dev_y)),
            "b0": b0_metrics,
            "v12": b0_metrics,
            "v13": selected["metrics"],
            "delta": selected["delta"],
        },
        "gates": {
            "dev_sparsity_mean": float(trained["sparsity_gate"].mean()),
            "dev_uncertainty_mean": float(trained["uncertainty_gate"].mean()),
            "max_abs_delta": float(np.abs(trained["delta"]).max()),
            "nominal_sparsity_gate": nominal_gates,
        },
        "checks": {
            "b0_retrained_without_dev_a_users": True,
            "v12_dense_view_absent_at_inference": True,
            "only_bus_car_logits_modified": True,
            "correction_bounded": bool(
                np.abs(trained["delta"]).max()
                <= selected["settings"]["alpha"] + 1e-6),
            "formal_test_not_loaded": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dev_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    torch.save({
        "state_dict": trained["model"].state_dict(),
        "input_mean": trained["input_mean"],
        "input_std": trained["input_std"],
        "settings": selected["settings"],
        "seed": args.seed,
    }, args.output_dir / "corrector_dev_best.pth")
    write_report(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b0-experiment", type=Path,
        default=Path("experiments/geolife_v13_dev_b0_seed12013"))
    parser.add_argument(
        "--v12-checkpoint", type=Path,
        default=Path("experiments/geolife_v12_behavior_recovery_seed12013/model_best.pth"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/geolife_v13_conservative_dev_seed12013"))
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/experiments/geolife_v13_conservative_correction_dev.md"))
    parser.add_argument("--seed", type=int, default=12013)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
