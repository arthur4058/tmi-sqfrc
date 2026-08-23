"""Train one validation-selected V23 short-trajectory expert variant."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

from scripts.train_multiscale_short_official import load_split
from scripts.train_multiscale_short_trajectory import ROOT, run_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--class-weight-power", type=float, required=True)
    parser.add_argument("--user-weight-power", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train = load_split("train")
    validation = load_split("val")
    clean = np.concatenate([train[0], validation[0]])
    noise = np.concatenate([train[1], validation[1]])
    mask = np.concatenate([train[2], validation[2]])
    labels = np.concatenate([train[3], validation[3]])
    users = np.concatenate([train[4], validation[4]])
    train_index = np.arange(len(train[3]))
    val_index = np.arange(len(train[3]), len(labels))
    output = ROOT / f"experiments/geolife_short_variants_v23_seed{args.seed}" / args.name
    output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(
        seed=args.seed,
        model="hybrid",
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        user_weight_power=args.user_weight_power,
        class_weight_power=args.class_weight_power,
        device=args.device,
    )
    result = run_fold(
        clean, noise, mask, labels, users, train_index, val_index,
        0, settings, output,
    )
    payload = {"name": args.name, "settings": vars(settings), "validation": result}
    payload["settings"]["device"] = str(payload["settings"]["device"])
    (output / "validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
