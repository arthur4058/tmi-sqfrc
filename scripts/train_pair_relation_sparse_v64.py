"""Multi-lag point-relation encoder for one sparse 60-second GPS segment."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_multiscale_short_trajectory import (
    DATA_ROOT, ROOT, normalize, normalization, seed_everything, tensorize,
    user_weights,
)


def load_split(split):
    base = DATA_ROOT / split
    clean_fs = np.load(base / "clean_multi_feature_segs.npy", allow_pickle=True)
    noisy_fs = np.load(base / "noise_multi_feature_segs.npy", allow_pickle=True)
    clean_xy = np.load(base / "clean_trj_segs.npy", allow_pickle=True)
    noisy_xy = np.load(base / "noise_trj_segs.npy", allow_pickle=True)
    labels = np.load(base / "clean_multi_feature_seg_labels.npy").astype(np.int64)
    users = np.load(base / "segment_user_ids.npy").astype(np.int64)
    clean, mask = tensorize(clean_fs, clean_xy)
    noisy, noisy_mask = tensorize(noisy_fs, noisy_xy)
    if not np.array_equal(mask, noisy_mask):
        raise ValueError("clean/noisy masks are not aligned")
    return clean, noisy, mask, labels, users


def metrics(probability, labels):
    prediction = probability.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


class PairRelationSparseNet(nn.Module):
    def __init__(self, channels=12, width=96, embedding=160, classes=5,
                 layers=2, dropout=.15):
        super().__init__()
        self.register_buffer("pair_i", torch.tensor(
            [i for i in range(6) for j in range(i + 1, 6)], dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(
            [j for i in range(6) for j in range(i + 1, 6)], dtype=torch.long))
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             norm=nn.LayerNorm(width))
        pooled = width * 3 + channels * 4 + 2
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled), nn.Linear(pooled, 256), nn.GELU(),
            nn.Dropout(.2), nn.Linear(256, embedding), nn.LayerNorm(embedding),
            nn.GELU())
        self.classifier = nn.Linear(embedding, classes)
        self.centres = nn.Parameter(torch.randn(classes, embedding) * .02)

    def forward(self, values, mask):
        raw = values.transpose(1, 2)
        point = self.point(raw) + self.point_type
        left, right = raw[:, self.pair_i], raw[:, self.pair_j]
        gap = ((self.pair_j - self.pair_i).float() / 5.0)
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
        quality = torch.stack((mask.sum(1) / 6.0, pair_mask.sum(1) / 15.0), 1)
        pooled = torch.cat((mean, variance.sqrt(), maximum, raw_mean,
                            raw_var.sqrt(), raw_min, raw_max, quality), 1)
        embedding = self.embedding(pooled)
        return self.classifier(embedding), embedding


def weighted(loss, weight):
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def evaluate(model, values, mask, labels, device):
    data = TensorDataset(torch.from_numpy(values), torch.from_numpy(mask))
    loader = DataLoader(data, batch_size=1024, shuffle=False, num_workers=0)
    output = []
    model.eval()
    with torch.no_grad():
        for x, m in loader:
            logits, _ = model(x.to(device), m.to(device))
            output.append(torch.softmax(logits, 1).cpu().numpy())
    probability = np.concatenate(output)
    return metrics(probability, labels), probability


def select_view(data, seed, mean, std):
    clean, noisy, mask, labels, users = data
    generator = random.Random(seed)
    choose_noisy = np.fromiter((generator.random() < .5 for _ in range(len(labels))),
                               dtype=bool, count=len(labels))
    values = clean.copy()
    values[choose_noisy] = noisy[choose_noisy]
    return normalize(values, mask, mean, std), mask, labels, users


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--name", default="relation_l2_w96")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--consistency-weight", type=float, default=.08)
    parser.add_argument("--centre-weight", type=float, default=.02)
    parser.add_argument("--class-weight-power", type=float, default=.35)
    parser.add_argument("--user-weight-power", type=float, default=.35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    train, validation = load_split("train"), load_split("val")
    mean, std = normalization(np.concatenate((train[0], train[1])),
                              np.concatenate((train[2], train[2])))
    clean = normalize(train[0], train[2], mean, std)
    noisy = normalize(train[1], train[2], mean, std)
    weights = user_weights(train[4], args.user_weight_power)
    data = TensorDataset(torch.from_numpy(clean), torch.from_numpy(noisy),
                         torch.from_numpy(train[2]), torch.from_numpy(train[3]),
                         torch.from_numpy(weights))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    val_x, val_mask, val_y, _ = select_view(validation, args.seed, mean, std)
    model = PairRelationSparseNet(width=args.width, layers=args.layers).to(args.device)
    counts = np.bincount(train[3], minlength=5).astype(np.float32)
    class_weights = counts ** (-args.class_weight_power)
    class_weights /= class_weights.mean()
    class_weights = torch.from_numpy(class_weights).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * .05)
    output_dir = ROOT / f"experiments/geolife_pair_relation_v64_seed{args.seed}/{args.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best, state, bad = None, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for clean_x, noisy_x, mask, labels, sample_weight in loader:
            clean_x, noisy_x = clean_x.to(args.device), noisy_x.to(args.device)
            mask, labels = mask.to(args.device), labels.to(args.device)
            sample_weight = sample_weight.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            clean_logits, clean_embedding = model(clean_x, mask)
            noisy_logits, noisy_embedding = model(noisy_x, mask)
            clean_ce = F.cross_entropy(clean_logits, labels, weight=class_weights,
                                       reduction="none", label_smoothing=.03)
            noisy_ce = F.cross_entropy(noisy_logits, labels, weight=class_weights,
                                       reduction="none", label_smoothing=.03)
            loss = .5 * (weighted(clean_ce, sample_weight)
                         + weighted(noisy_ce, sample_weight))
            clean_log, noisy_log = (F.log_softmax(clean_logits, 1),
                                    F.log_softmax(noisy_logits, 1))
            consistency = .5 * (
                F.kl_div(clean_log, noisy_log.exp(), reduction="batchmean")
                + F.kl_div(noisy_log, clean_log.exp(), reduction="batchmean"))
            alignment = 1.0 - F.cosine_similarity(clean_embedding,
                                                   noisy_embedding, dim=1)
            centre = F.normalize(model.centres, dim=1)[labels]
            metric_loss = .5 * (
                1.0 - F.cosine_similarity(F.normalize(clean_embedding, dim=1), centre)
                + 1.0 - F.cosine_similarity(F.normalize(noisy_embedding, dim=1), centre))
            loss = (loss + args.consistency_weight * consistency
                    + args.consistency_weight * weighted(alignment, sample_weight)
                    + args.centre_weight * weighted(metric_loss, sample_weight))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite V64 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
        current, _ = evaluate(model, val_x, val_mask, val_y, args.device)
        rank = (min(current.values()), sum(current.values()))
        if best is None or rank > best[0]:
            best = (rank, epoch, current)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"epoch={epoch:03d} acc={current['accuracy']:.5f} "
              f"f1={current['macro_f1']:.5f} best={best[1]:03d}", flush=True)
        if bad >= args.patience:
            break
    checkpoint = output_dir / "model_best.pth"
    torch.save({"state_dict": state, "mean": mean, "std": std,
                "epoch": best[1], "settings": vars(args)}, checkpoint)
    model.load_state_dict(state)
    _, expert_val = evaluate(model, val_x, val_mask, val_y, args.device)
    saved = np.load(ROOT / f"experiments/geolife_nohistory_v58_seed{args.seed}/v53_predictions.npz")
    base_val, base_score = saved["validation"], metrics(saved["validation"], val_y)
    trials = []
    for weight in np.linspace(0, .7, 71):
        fused = (1-weight)*base_val + weight*expert_val
        current = metrics(fused, val_y)
        delta = {key: current[key] - base_score[key] for key in base_score}
        trials.append({"weight": float(weight), "metrics": current, "delta": delta})
    trials.sort(key=lambda row: (min(row["delta"].values()),
                                 sum(row["delta"].values()), -row["weight"]), reverse=True)
    winner = trials[0]
    result = {
        "protocol": "V64-current-segment-multi-lag-relation-encoder",
        "settings": vars(args), "best_epoch": best[1],
        "expert_validation": best[2], "v53_validation": base_score,
        "fusion": winner,
        "test_gate": "both validation metrics must improve by >=0.003",
    }
    if min(winner["delta"].values()) >= .003:
        test_x, test_mask, test_y, _ = select_view(load_split("test"), args.seed,
                                                   mean, std)
        _, expert_test = evaluate(model, test_x, test_mask, test_y, args.device)
        fused = (1-winner["weight"])*saved["test"] + winner["weight"]*expert_test
        result["test"] = {
            "v53": metrics(saved["test"], test_y),
            "expert": metrics(expert_test, test_y),
            "v64": metrics(fused, test_y),
        }
        result["test"]["delta"] = {
            key: result["test"]["v64"][key] - result["test"]["v53"][key]
            for key in ("accuracy", "macro_f1")}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
