"""Run a rate-generalized V74 relation expert against matched B0 models.

The original V74 implementation was intentionally specialized to the 60-second
view (at most six observations) and fused with a 60-second-only V53 cache.  This
entrypoint keeps the V74 observation-drop objective, uses at most 16 uniformly
spaced real observations for the relation expert, and fuses it directly with
the matched B0 checkpoint selected for the same rate and seed.  B0 still sees
the complete input sequence; anchor selection affects only the added expert.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import main as training_main
from scripts.evaluate_dual_expert_consensus import choose_temperature, softmax
from scripts.train_multiscale_short_trajectory import (
    finite,
    normalize,
    normalization,
    seed_everything,
    signed_log,
    user_weights,
)
from tmi.datasets import dataset


ROOT = Path(__file__).resolve().parents[1]
RATES = (5, 10, 20, 30, 60)
FEATURE_INDICES = (2, 3, 4, 5, 7, 8)
CLASS_NAMES = ("Walk", "Bike", "Bus", "Car", "Train")


def anchor_count(rate: int) -> int:
    """Maximum real observations used by the relation expert."""
    if rate not in RATES:
        raise ValueError(f"unsupported rate: {rate}")
    return min(16, math.ceil(300 / rate) + 1)


def anchor_indices(length: int, maximum: int) -> np.ndarray:
    """Select real points spanning the whole window without interpolation."""
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    indices = np.rint(np.linspace(0, length - 1, maximum)).astype(np.int64)
    if len(np.unique(indices)) != maximum:
        raise RuntimeError("uniform anchor selection produced duplicates")
    return indices


def tensorize(feature_samples, trajectory_samples, maximum: int):
    output = np.zeros((len(feature_samples), 12, maximum), dtype=np.float32)
    mask = np.zeros((len(feature_samples), maximum), dtype=np.float32)
    for row in range(len(feature_samples)):
        length = min(
            len(finite(feature_samples[row, FEATURE_INDICES[0]]).reshape(-1)),
            len(finite(trajectory_samples[row, 0]).reshape(-1)),
            len(finite(trajectory_samples[row, 1]).reshape(-1)),
        )
        if length == 0:
            continue
        indices = anchor_indices(length, maximum)
        used = len(indices)
        mask[row, :used] = 1.0
        robust_channels = [
            finite(feature_samples[row, index]).reshape(-1)[:length][indices]
            for index in FEATURE_INDICES
        ]
        heading = finite(feature_samples[row, 6]).reshape(-1)[:length][indices]
        heading_radians = np.deg2rad(heading)
        latitude = finite(trajectory_samples[row, 0]).reshape(-1)[:length][indices]
        longitude = finite(trajectory_samples[row, 1]).reshape(-1)[:length][indices]
        mean_latitude = np.deg2rad(float(latitude.mean()))
        relative_y = (latitude - latitude[0]) * 111_320.0
        relative_x = (
            (longitude - longitude[0]) * 111_320.0 * math.cos(mean_latitude)
        )
        step_x = np.diff(relative_x, prepend=relative_x[0])
        step_y = np.diff(relative_y, prepend=relative_y[0])
        robust_channels.extend((relative_x, relative_y, step_x, step_y))
        output[row, :10, :used] = signed_log(np.stack(robust_channels, axis=0))
        output[row, 10, :used] = np.sin(heading_radians)
        output[row, 11, :used] = np.cos(heading_radians)
    return output, mask


def load_split(rate: int, split: str):
    base = ROOT / f"data/geolife_five_rate_fixed_{rate}s_features/{split}"
    clean_fs = np.load(base / "clean_multi_feature_segs.npy", allow_pickle=True)
    noisy_fs = np.load(base / "noise_multi_feature_segs.npy", allow_pickle=True)
    clean_xy = np.load(base / "clean_trj_segs.npy", allow_pickle=True)
    noisy_xy = np.load(base / "noise_trj_segs.npy", allow_pickle=True)
    labels = np.load(base / "clean_multi_feature_seg_labels.npy").astype(np.int64)
    users = np.load(base / "segment_user_ids.npy").astype(np.int64)
    maximum = anchor_count(rate)
    clean, mask = tensorize(clean_fs, clean_xy, maximum)
    noisy, noisy_mask = tensorize(noisy_fs, noisy_xy, maximum)
    if not np.array_equal(mask, noisy_mask):
        raise ValueError("clean/noisy masks are not aligned")
    return clean, noisy, mask, labels, users


def all_pairs(maximum: int):
    left, right = [], []
    for i in range(maximum):
        for j in range(i + 1, maximum):
            left.append(i)
            right.append(j)
    return np.asarray(left), np.asarray(right)


class RateGeneralizedRelationNet(nn.Module):
    def __init__(self, maximum: int, channels=12, width=96, embedding=160,
                 classes=5, layers=2, dropout=.15):
        super().__init__()
        pair_i, pair_j = all_pairs(maximum)
        self.maximum = maximum
        self.register_buffer("pair_i", torch.from_numpy(pair_i).long())
        self.register_buffer("pair_j", torch.from_numpy(pair_j).long())
        self.point = nn.Sequential(nn.Linear(channels, width), nn.LayerNorm(width),
                                   nn.GELU())
        self.relation = nn.Sequential(
            nn.Linear(channels * 3 + 1, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU())
        self.point_type = nn.Parameter(torch.zeros(1, 1, width))
        self.relation_type = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 3,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(width))
        pooled = width * 3 + channels * 4 + 2
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled), nn.Linear(pooled, 256), nn.GELU(),
            nn.Dropout(.2), nn.Linear(256, embedding), nn.LayerNorm(embedding),
            nn.GELU())
        self.classifier = nn.Linear(embedding, classes)

    def forward(self, values, mask):
        raw = values.transpose(1, 2)
        point = self.point(raw) + self.point_type
        left, right = raw[:, self.pair_i], raw[:, self.pair_j]
        denominator = max(1, self.maximum - 1)
        gap = (self.pair_j - self.pair_i).float() / float(denominator)
        gap = gap[None, :, None].expand(len(raw), -1, -1)
        relation_raw = torch.cat((left, right, right - left, gap), 2)
        relation = self.relation(relation_raw) + self.relation_type
        pair_mask = mask[:, self.pair_i] * mask[:, self.pair_j]
        token = torch.cat((point, relation), 1)
        token_mask = torch.cat((mask, pair_mask), 1)
        token = self.encoder(token, src_key_padding_mask=~token_mask.bool())
        valid = token_mask[:, :, None]
        count = valid.sum(1).clamp_min(1.0)
        mean = (token * valid).sum(1) / count
        variance = ((token - mean[:, None]) ** 2 * valid).sum(1) / count
        maximum = token.masked_fill(~valid.bool(), -torch.inf).max(1).values
        raw_valid = mask[:, :, None]
        raw_count = raw_valid.sum(1).clamp_min(1.0)
        raw_mean = (raw * raw_valid).sum(1) / raw_count
        raw_var = ((raw - raw_mean[:, None]) ** 2 * raw_valid).sum(1) / raw_count
        raw_min = raw.masked_fill(~raw_valid.bool(), torch.inf).min(1).values
        raw_max = raw.masked_fill(~raw_valid.bool(), -torch.inf).max(1).values
        quality = torch.stack((
            mask.sum(1) / float(self.maximum),
            pair_mask.sum(1) / float(len(self.pair_i)),
        ), 1)
        pooled = torch.cat((mean, variance.sqrt(), maximum, raw_mean,
                            raw_var.sqrt(), raw_min, raw_max, quality), 1)
        embedding = self.embedding(pooled)
        return self.classifier(embedding), embedding


def weighted(loss, weight):
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def point_drop(mask, probability):
    result = mask.clone()
    apply = torch.rand(len(mask), device=mask.device) < probability
    lengths = mask.sum(1).long()
    for row in torch.nonzero(apply & (lengths > 3), as_tuple=False).flatten():
        index = torch.randint(1, int(lengths[row]) - 1, (1,), device=mask.device)
        result[row, index] = 0.0
    return result


def select_aligned(data, seed, mean, std):
    clean, noisy, mask, labels, users = data
    generator = random.Random(seed)
    choose_noisy = np.fromiter(
        (generator.random() < .5 for _ in range(len(labels))),
        dtype=bool, count=len(labels))
    values = clean.copy()
    values[choose_noisy] = noisy[choose_noisy]
    return normalize(values, mask, mean, std), mask, labels, users


def metric_summary(probability, labels):
    prediction = probability.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "per_class_f1": {
            name: float(value) for name, value in zip(
                CLASS_NAMES,
                f1_score(labels, prediction, labels=np.arange(5), average=None),
            )
        },
    }


def relation_logits(model, values, mask, device):
    data = TensorDataset(torch.from_numpy(values), torch.from_numpy(mask))
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=0)
    logits = []
    model.eval()
    with torch.inference_mode():
        for x, m in loader:
            current, _ = model(x.to(device), m.to(device))
            logits.append(current.cpu().numpy())
    return np.concatenate(logits)


def b0_logits(rate: int, seed: int, split: str, device: str):
    base = ROOT / f"experiments/geolife_matched_b0_fixed_{rate}s_seed{seed}"
    config = json.loads((base / "configuration.json").read_text(encoding="utf-8"))
    output = ROOT / f"experiments/geolife_v74_generalized_{rate}s_seed{seed}/b0_eval"
    output.mkdir(parents=True, exist_ok=True)
    config.update({
        "task": "dual_branch_classification",
        "test_only": "testset",
        "load_model": str(base / "checkpoints/model_best.pth"),
        "output_dir": str(output),
        "records_file": str(output / "records.xlsx"),
        "gpu": "0" if device.startswith("cuda") else "-1",
    })
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()
    loader = pipeline.val_loader if split == "val" else pipeline.test_loader
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logits, labels = [], []
    pipeline.model.eval()
    with torch.inference_mode():
        for batch in loader:
            x1, x2, mask1, mask2, target = batch[:5]
            current = pipeline.model(
                x1.to(pipeline.device), mask1.to(pipeline.device),
                x2.to(pipeline.device), mask2.to(pipeline.device))
            logits.append(current.cpu().numpy())
            labels.append(target.reshape(-1).cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def choose_fusion(base_probability, expert_probability, labels):
    base_score = metric_summary(base_probability, labels)
    candidates = []
    for weight in np.linspace(0.0, 0.7, 71):
        fused = (1.0 - weight) * base_probability + weight * expert_probability
        current = metric_summary(fused, labels)
        delta = {
            key: current[key] - base_score[key]
            for key in ("accuracy", "macro_f1")
        }
        rank = (
            min(delta.values()),
            sum(delta.values()),
            current["macro_f1"],
            current["accuracy"],
            -float(weight),
        )
        candidates.append((rank, float(weight), current, delta))
    _, weight, current, delta = max(candidates, key=lambda row: row[0])
    return {"weight": weight, "metrics": current, "delta": delta}


def train_one(rate: int, seed: int, args):
    output = ROOT / f"experiments/geolife_v74_generalized_{rate}s_seed{seed}"
    result_path = output / "result.json"
    if args.skip_existing and result_path.exists():
        print(f"SKIP existing {result_path}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    train = load_split(rate, "train")
    validation = load_split(rate, "val")
    mean, std = normalization(
        np.concatenate((train[0], train[1])),
        np.concatenate((train[2], train[2])))
    clean = normalize(train[0], train[2], mean, std)
    noisy = normalize(train[1], train[2], mean, std)
    sample_weights = user_weights(train[4], .35)
    train_data = TensorDataset(
        torch.from_numpy(clean), torch.from_numpy(noisy),
        torch.from_numpy(train[2]), torch.from_numpy(train[3]),
        torch.from_numpy(sample_weights))
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    val_x, val_mask, val_y, _ = select_aligned(
        validation, seed, mean, std)
    model = RateGeneralizedRelationNet(
        anchor_count(rate), width=args.width, layers=args.layers).to(args.device)
    counts = np.bincount(train[3], minlength=5).astype(np.float32)
    class_weights = counts ** (-.35)
    class_weights /= class_weights.mean()
    class_weights = torch.from_numpy(class_weights).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * .05)
    best, state, bad = None, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for clean_x, noisy_x, mask, labels, sample_weight in loader:
            clean_x, noisy_x = clean_x.to(args.device), noisy_x.to(args.device)
            mask, labels = mask.to(args.device), labels.to(args.device)
            sample_weight = sample_weight.to(args.device)
            dropped_mask = point_drop(mask, args.drop_probability)
            optimizer.zero_grad(set_to_none=True)
            clean_logits, _ = model(clean_x, mask)
            noisy_logits, noisy_embedding = model(noisy_x, mask)
            drop_logits, drop_embedding = model(noisy_x, dropped_mask)
            clean_ce = F.cross_entropy(
                clean_logits, labels, weight=class_weights,
                reduction="none", label_smoothing=.03)
            noisy_ce = F.cross_entropy(
                noisy_logits, labels, weight=class_weights,
                reduction="none", label_smoothing=.03)
            drop_ce = F.cross_entropy(
                drop_logits, labels, weight=class_weights,
                reduction="none", label_smoothing=.03)
            loss = .5 * (
                weighted(clean_ce, sample_weight)
                + weighted(noisy_ce, sample_weight)
            ) + args.drop_weight * weighted(drop_ce, sample_weight)
            target = .5 * (
                torch.softmax(clean_logits.detach(), 1)
                + torch.softmax(noisy_logits.detach(), 1))
            consistency = F.kl_div(
                F.log_softmax(drop_logits, 1), target, reduction="batchmean")
            alignment = 1.0 - F.cosine_similarity(
                drop_embedding, noisy_embedding.detach(), dim=1)
            loss = (
                loss + args.consistency_weight * consistency
                + args.consistency_weight * weighted(alignment, sample_weight))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite generalized V74 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
        val_logits = relation_logits(model, val_x, val_mask, args.device)
        current = metric_summary(softmax(val_logits), val_y)
        rank = (min(current["accuracy"], current["macro_f1"]),
                current["accuracy"] + current["macro_f1"])
        if best is None or rank > best[0]:
            best = (rank, epoch, current)
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
        print(
            f"rate={rate:02d} seed={seed} epoch={epoch:03d} "
            f"acc={current['accuracy']:.5f} f1={current['macro_f1']:.5f} "
            f"best={best[1]:03d}", flush=True)
        if bad >= args.patience:
            break

    model.load_state_dict(state)
    val_expert_logits = relation_logits(model, val_x, val_mask, args.device)
    test_x, test_mask, test_y, _ = select_aligned(
        load_split(rate, "test"), seed, mean, std)
    test_expert_logits = relation_logits(model, test_x, test_mask, args.device)

    val_b0_logits, val_b0_y = b0_logits(rate, seed, "val", args.device)
    test_b0_logits, test_b0_y = b0_logits(rate, seed, "test", args.device)
    if not np.array_equal(val_y, val_b0_y):
        raise ValueError("V74/B0 validation labels are not aligned")
    if not np.array_equal(test_y, test_b0_y):
        raise ValueError("V74/B0 test labels are not aligned")
    b0_temperature = choose_temperature(val_b0_logits, val_y)
    expert_temperature = choose_temperature(val_expert_logits, val_y)
    val_b0 = softmax(val_b0_logits, b0_temperature)
    val_expert = softmax(val_expert_logits, expert_temperature)
    test_b0 = softmax(test_b0_logits, b0_temperature)
    test_expert = softmax(test_expert_logits, expert_temperature)
    fusion = choose_fusion(val_b0, val_expert, val_y)
    weight = fusion["weight"]
    test_v74 = (1.0 - weight) * test_b0 + weight * test_expert

    result = {
        "protocol": "geolife-v74-rate-generalized-current-segment-v1",
        "rate_seconds": rate,
        "seed": seed,
        "anchors": anchor_count(rate),
        "uses_interpolation": False,
        "b0_checkpoint": str(
            ROOT / f"experiments/geolife_matched_b0_fixed_{rate}s_seed{seed}"
            / "checkpoints/model_best.pth"),
        "settings": {
            "width": args.width,
            "layers": args.layers,
            "drop_probability": args.drop_probability,
            "drop_weight": args.drop_weight,
            "consistency_weight": args.consistency_weight,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "best_epoch": best[1],
        "temperatures": {
            "b0": float(b0_temperature),
            "relation_expert": float(expert_temperature),
        },
        "validation": {
            "b0": metric_summary(val_b0, val_y),
            "expert": metric_summary(val_expert, val_y),
            "fusion": fusion,
        },
        "test": {
            "b0": metric_summary(test_b0, test_y),
            "expert": metric_summary(test_expert, test_y),
            "v74": metric_summary(test_v74, test_y),
        },
    }
    result["test"]["delta"] = {
        key: result["test"]["v74"][key] - result["test"]["b0"][key]
        for key in ("accuracy", "macro_f1")
    }
    torch.save({
        "state_dict": state,
        "mean": mean,
        "std": std,
        "rate_seconds": rate,
        "seed": seed,
        "settings": result["settings"],
    }, output / "model_best.pth")
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def experiment_grid(five_rate: bool, multiseed: bool):
    grid = []
    if five_rate:
        grid.extend((rate, 10086) for rate in RATES)
    if multiseed:
        grid.extend((rate, seed) for rate in (30, 60) for seed in (42, 2024))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--five-rate", action="store_true")
    parser.add_argument("--multiseed", action="store_true")
    parser.add_argument("--rate", type=int, choices=RATES)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--drop-probability", type=float, default=.5)
    parser.add_argument("--drop-weight", type=float, default=.35)
    parser.add_argument("--consistency-weight", type=float, default=.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.rate is not None:
        grid = [(args.rate, args.seed)]
    else:
        grid = experiment_grid(args.five_rate, args.multiseed)
    if not grid:
        parser.error("select --rate or at least one of --five-rate/--multiseed")
    results = [train_one(rate, seed, args) for rate, seed in grid]
    summary = ROOT / "reports/experiments/geolife_v74_five_rate_multiseed.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"WROTE {summary}")


if __name__ == "__main__":
    main()
