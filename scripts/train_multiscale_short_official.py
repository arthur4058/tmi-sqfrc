"""Frozen official validation/test for the multi-scale short-trajectory net."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from scripts.train_multiscale_short_trajectory import (
    DATA_ROOT,
    ROOT,
    MultiScaleShortTrajectoryNet,
    evaluate,
    make_loader,
    normalize,
    run_fold,
    tensorize,
)


def load_split(split):
    base = DATA_ROOT / split
    clean_features = np.load(
        base / "clean_multi_feature_segs.npy", allow_pickle=True
    )
    noise_features = np.load(
        base / "noise_multi_feature_segs.npy", allow_pickle=True
    )
    clean_trajectories = np.load(base / "clean_trj_segs.npy", allow_pickle=True)
    noise_trajectories = np.load(base / "noise_trj_segs.npy", allow_pickle=True)
    labels = np.load(base / "clean_multi_feature_seg_labels.npy").astype(np.int64)
    user_path = base / "segment_user_ids.npy"
    users = (
        np.load(user_path).astype(np.int64)
        if user_path.exists()
        else np.arange(len(labels), dtype=np.int64)
    )
    clean, mask = tensorize(clean_features, clean_trajectories)
    noise, noise_mask = tensorize(noise_features, noise_trajectories)
    if not np.array_equal(mask, noise_mask):
        raise ValueError(f"Clean/noise masks are not aligned for {split}")
    return clean, noise, mask, labels, users


def select_half_noise(clean, noise, seed):
    generator = random.Random(int(seed))
    use_noise = np.fromiter(
        (generator.random() < 0.5 for _ in range(len(clean))),
        dtype=bool, count=len(clean),
    )
    values = clean.copy()
    values[use_noise] = noise[use_noise]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_dir = ROOT / f"experiments/geolife_multiscale_short_official_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    validation = load_split("val")
    clean = np.concatenate([train[0], validation[0]])
    noise = np.concatenate([train[1], validation[1]])
    mask = np.concatenate([train[2], validation[2]])
    labels = np.concatenate([train[3], validation[3]])
    users = np.concatenate([train[4], validation[4]])
    train_index = np.arange(len(train[3]))
    val_index = np.arange(len(train[3]), len(labels))
    settings = SimpleNamespace(
        seed=args.seed,
        model="hybrid",
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        user_weight_power=0.5,
        class_weight_power=1.0,
        device=args.device,
    )
    validation_result = run_fold(
        clean, noise, mask, labels, users, train_index, val_index,
        0, settings, output_dir,
    )

    checkpoint_path = output_dir / "fold0_hybrid_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultiScaleShortTrajectoryNet(
        kernels=tuple(checkpoint["kernels"]), summary_pool=True
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device)

    test_clean, test_noise, test_mask, test_labels, test_users = load_split("test")
    test_values = select_half_noise(test_clean, test_noise, args.seed)
    test_values = normalize(
        test_values, test_mask, checkpoint["mean"], checkpoint["std"]
    )
    test_loader = make_loader(
        test_values, test_mask, test_labels,
        np.ones(len(test_labels), dtype=np.float32), args.batch_size, False,
    )
    test_result = evaluate(model, test_loader, args.device)
    predictions = []
    model.eval()
    with torch.no_grad():
        for values, batch_mask, _, _ in test_loader:
            logits = model(values.to(args.device), batch_mask.to(args.device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predictions)
    result = {
        "protocol": "geolife-fixed-60s-official-user-disjoint-v1",
        "seed": args.seed,
        "model": "hybrid_multiscale_short_trajectory",
        "selection": "official validation only; test evaluated once after freeze",
        "training": vars(settings),
        "validation": validation_result,
        "test": test_result,
        "test_users": int(len(np.unique(test_users))),
        "test_samples": int(len(test_labels)),
        "confusion_matrix": confusion_matrix(test_labels, predictions).tolist(),
        "classification_report": classification_report(
            test_labels, predictions, output_dict=True, zero_division=0
        ),
    }
    result["training"]["device"] = str(result["training"]["device"])
    (output_dir / "official_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
