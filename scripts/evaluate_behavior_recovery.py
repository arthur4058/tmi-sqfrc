#!/usr/bin/env python3
"""Train V12 on Train-A and evaluate physical behavior recovery on Dev-A."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmi.models.behavior_recovery import IntervalBehaviorRecovery

TARGET_NAMES = (
    "mean_speed", "std_speed", "max_speed", "stop_ratio",
    "acceleration_energy", "turning_energy")
KEY_TARGETS = ("mean_speed", "max_speed", "stop_ratio")
LOG_TARGET_INDICES = (0, 1, 2, 4, 5)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_valid(path: Path):
    arrays = np.load(path, allow_pickle=True)
    valid = arrays["valid_targets"].astype(bool)
    return (
        arrays["sparse_features"][valid].astype(np.float32),
        arrays["behavior_targets"][valid].astype(np.float32),
    )


def transform_targets(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[:, LOG_TARGET_INDICES] = np.log1p(
        np.maximum(result[:, LOG_TARGET_INDICES], 0.0))
    return result


def inverse_targets(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[:, LOG_TARGET_INDICES] = np.expm1(result[:, LOG_TARGET_INDICES])
    result[:, LOG_TARGET_INDICES] = np.maximum(
        result[:, LOG_TARGET_INDICES], 0.0)
    result[:, 3] = np.clip(result[:, 3], 0.0, 1.0)
    return result


def normalize(values, mean, std):
    return (values - mean) / std


def predict(model, features, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def metrics(y_true, predictions):
    mae = np.mean(np.abs(y_true - predictions), axis=0)
    r2 = []
    pearson = []
    for index in range(y_true.shape[1]):
        residual = np.sum((y_true[:, index] - predictions[:, index]) ** 2)
        total = np.sum((y_true[:, index] - y_true[:, index].mean()) ** 2)
        r2.append(float(1.0 - residual / total) if total > 0 else 0.0)
        if np.std(predictions[:, index]) < 1e-12 or np.std(y_true[:, index]) < 1e-12:
            pearson.append(0.0)
        else:
            pearson.append(float(np.corrcoef(
                y_true[:, index], predictions[:, index])[0, 1]))
    return mae, np.asarray(r2), np.asarray(pearson)


def direct_observed(features):
    result = np.zeros((len(features), len(TARGET_NAMES)), dtype=np.float32)
    result[:, 0] = features[:, 2]
    result[:, 2] = features[:, 2]
    result[:, 3] = (features[:, 1] < 0.5).astype(np.float32)
    result[:, 5] = features[:, 4]
    return result


def write_report(path: Path, result: dict):
    lines = [
        "# GeoLife V12 行为恢复 Dev-A 验证", "",
        "## 结论", "",
        f"**V12 {result['decision']}**。", "",
        "本阶段只使用原五档 training users 内部重新划分的 Train-A/Dev-A，",
        "没有加载原 validation 或正式 test。5 秒视图只用于构造训练监督，",
        "V12 输入仅包含 60 秒稀疏区间可观测特征。", "",
        "## 恢复结果", "",
        "| Target | 常数 MAE | 稀疏直接观测 MAE | V12 MAE | 相对常数改善 | R² | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in TARGET_NAMES:
        item = result["targets"][name]
        lines.append(
            f"| {name} | {item['constant_mae']:.6f} | "
            f"{item['direct_mae']:.6f} | {item['v12_mae']:.6f} | "
            f"{item['improvement_vs_constant'] * 100:.2f}% | "
            f"{item['r2']:.4f} | {item['pearson']:.4f} |")
    lines.extend([
        "", "## 训练状态", "",
        f"- 最佳 epoch：{result['training']['best_epoch']}",
        f"- 最佳 Dev normalized MAE：{result['training']['best_dev_normalized_mae']:.6f}",
        f"- 对应 Train normalized MAE：{result['training']['train_normalized_mae']:.6f}",
        f"- Train/Dev gap ratio：{result['training']['train_dev_gap_ratio']:.4f}",
        f"- 新增参数：{result['training']['trainable_parameters']}",
        f"- 训练时间：{result['training']['runtime_seconds']:.2f} 秒",
        "", "## PASS 规则", "",
        "- mean_speed、max_speed、stop_ratio 至少两个相对常数预测器改善 ≥10%；",
        "- 六个目标中至少四个优于常数预测器；",
        "- Train/Dev normalized MAE 比值不超过 2，避免严重分离。", "",
        f"关键目标通过数：{result['checks']['key_targets_passed']}/3。",
        f"全部目标优于常数数：{result['checks']['targets_beating_constant']}/6。",
        f"无严重 Train/Dev 分离：{result['checks']['no_severe_train_dev_gap']}。", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_x, train_y = load_valid(args.data_dir / "train_a.npz")
    dev_x, dev_y = load_valid(args.data_dir / "dev_a.npz")

    input_mean = train_x.mean(axis=0)
    input_std = train_x.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    train_x_norm = np.clip(normalize(train_x, input_mean, input_std), -10.0, 10.0).astype(np.float32)
    dev_x_norm = np.clip(normalize(dev_x, input_mean, input_std), -10.0, 10.0).astype(np.float32)

    train_y_transformed = transform_targets(train_y)
    target_mean = train_y_transformed.mean(axis=0)
    target_std = train_y_transformed.std(axis=0)
    target_std[target_std < 1e-8] = 1.0
    train_y_norm = normalize(train_y_transformed, target_mean, target_std).astype(np.float32)
    dev_y_norm = normalize(transform_targets(dev_y), target_mean, target_std).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda")
    model = IntervalBehaviorRecovery(
        input_dim=train_x.shape[1], hidden_dim=args.hidden_dim,
        output_dim=train_y.shape[1], dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    best_state = None
    best_dev = math.inf
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x.to(device, non_blocking=True))
            loss = loss_fn(output, batch_y.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite V12 loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        dev_norm_prediction = predict(
            model, dev_x_norm, args.batch_size * 2, device)
        dev_score = float(np.mean(np.abs(dev_y_norm - dev_norm_prediction)))
        if dev_score < best_dev - 1e-5:
            best_dev = dev_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch:03d} dev_normalized_mae={dev_score:.6f} best={best_dev:.6f}")
        if stale >= args.patience:
            break
    runtime = time.time() - start
    model.load_state_dict(best_state)

    train_pred_norm = predict(model, train_x_norm, args.batch_size * 2, device)
    dev_pred_norm = predict(model, dev_x_norm, args.batch_size * 2, device)
    train_norm_mae = float(np.mean(np.abs(train_y_norm - train_pred_norm)))
    dev_transformed = dev_pred_norm * target_std + target_mean
    dev_prediction = inverse_targets(dev_transformed)

    constant_prediction = np.broadcast_to(train_y.mean(axis=0), dev_y.shape)
    direct_prediction = direct_observed(dev_x)
    constant_mae, _, _ = metrics(dev_y, constant_prediction)
    direct_mae, _, _ = metrics(dev_y, direct_prediction)
    v12_mae, v12_r2, v12_pearson = metrics(dev_y, dev_prediction)
    improvement = (constant_mae - v12_mae) / np.maximum(constant_mae, 1e-12)
    key_passed = sum(
        improvement[TARGET_NAMES.index(name)] >= 0.10 for name in KEY_TARGETS)
    targets_beating = int(np.count_nonzero(v12_mae < constant_mae))
    gap_ratio = float(best_dev / max(train_norm_mae, 1e-12))
    no_gap = gap_ratio <= 2.0
    passed = key_passed >= 2 and targets_beating >= 4 and no_gap

    target_results = {}
    for index, name in enumerate(TARGET_NAMES):
        target_results[name] = {
            "constant_mae": float(constant_mae[index]),
            "direct_mae": float(direct_mae[index]),
            "v12_mae": float(v12_mae[index]),
            "improvement_vs_constant": float(improvement[index]),
            "r2": float(v12_r2[index]),
            "pearson": float(v12_pearson[index]),
        }
    result = {
        "protocol": "geolife-v12-behavior-recovery-dev-v1",
        "decision": "PASS" if passed else "FAIL",
        "formal_validation_or_test_loaded": False,
        "seed": args.seed,
        "device": str(device),
        "train_intervals": int(len(train_x)),
        "dev_intervals": int(len(dev_x)),
        "targets": target_results,
        "checks": {
            "key_targets_passed": int(key_passed),
            "targets_beating_constant": targets_beating,
            "no_severe_train_dev_gap": bool(no_gap),
            "dense_view_absent_from_model_input": True,
            "normalization_fit_on_train_a_only": True,
            "formal_test_not_loaded": True,
        },
        "training": {
            "best_epoch": best_epoch,
            "best_dev_normalized_mae": best_dev,
            "train_normalized_mae": train_norm_mae,
            "train_dev_gap_ratio": gap_ratio,
            "runtime_seconds": runtime,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_log_indices": LOG_TARGET_INDICES,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, args.output_dir / "model_best.pth")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/v12_behavior_targets"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/geolife_v12_behavior_recovery_seed12013"))
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/geolife_v12_behavior_recovery_dev.md"))
    parser.add_argument("--seed", type=int, default=12013)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
