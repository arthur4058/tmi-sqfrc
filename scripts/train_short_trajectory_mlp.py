"""Train a location-invariant MLP expert for extremely short GPS segments."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_dual_expert_consensus import metrics, softmax
from scripts.train_statistical_motion_expert import (
    choose_mix,
    load_split,
    load_v2_validation,
)


ROOT = Path(__file__).resolve().parents[1]


class ShortTrajectoryMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 5):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.15),
        )
        self.residual = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(256, 256),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, n_classes),
        )

    def forward(self, values):
        hidden = self.input(values)
        hidden = hidden + 0.5 * self.residual(hidden)
        return self.output(hidden)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probabilities(model, values, mean, scale, device, batch_size=4096):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = (values[start:start + batch_size] - mean) / scale
            logits = model(torch.from_numpy(batch).float().to(device))
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def train(seed: int = 10086):
    os.chdir(ROOT)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / f"experiments/geolife_short_mlp_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_split("train", seed, training=True)
    val_x, val_y = load_split("val", seed, training=False)
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    scale = train_x.std(axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = ((train_x - mean) / scale).astype(np.float32)

    counts = np.bincount(train_y, minlength=5).astype(np.float32)
    class_weight = np.power(counts.max() / counts, 0.25)
    model = ShortTrajectoryMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=2e-5)
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weight).to(device),
        label_smoothing=0.03,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized),
                      torch.from_numpy(train_y)),
        batch_size=1024, shuffle=True, generator=loader_generator,
        num_workers=0, pin_memory=True,
    )

    best = None
    stale = 0
    history = []
    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        scheduler.step()
        val_probability = probabilities(model, val_x, mean, scale, device)
        score = metrics(val_probability, val_y)
        record = {"epoch": epoch,
                  "loss": total_loss / len(train_y), **score}
        history.append(record)
        rank = (min(score["accuracy"], score["macro_f1"]),
                score["accuracy"] + score["macro_f1"])
        if best is None or rank > best[0]:
            best = (rank, epoch, {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}, score)
            stale = 0
        else:
            stale += 1
        print(json.dumps(record), flush=True)
        if stale >= 10:
            break

    _, best_epoch, state, standalone = best
    model.load_state_dict(state)
    val_probability = probabilities(model, val_x, mean, scale, device)
    v2_probability, v2_targets, temperatures, v2_weights = (
        load_v2_validation(seed)
    )
    if not np.array_equal(val_y, v2_targets):
        raise ValueError("MLP and V2 validation targets are not aligned")
    weight, v2_score, fusion_score = choose_mix(
        v2_probability, val_probability, val_y
    )
    delta = {key: fusion_score[key] - v2_score[key]
             for key in ("accuracy", "macro_f1")}
    checkpoint = {
        "state_dict": state,
        "mean": torch.from_numpy(mean),
        "scale": torch.from_numpy(scale),
        "input_dim": train_x.shape[1],
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_dir / "model_best.pth")
    result = {
        "protocol": "geolife-short-trajectory-mlp-validation-v1",
        "seed": seed,
        "sampling_interval_seconds": 60,
        "location_and_timestamp_invariant": True,
        "best_epoch": best_epoch,
        "standalone_validation": standalone,
        "v2_validation": v2_score,
        "fusion_weight": weight,
        "fusion_validation": fusion_score,
        "delta_vs_v2": delta,
        "accepted_for_exploratory_test": (
            delta["accuracy"] >= 0.01 and delta["macro_f1"] >= 0.01
        ),
        "v2_temperatures": temperatures,
        "v2_weights": [float(value) for value in v2_weights],
        "history": history,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    train(args.seed)


if __name__ == "__main__":
    main()
