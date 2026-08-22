"""Evaluate user-balanced HGB on training-only user-disjoint folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from scripts.cross_validate_statistical_experts_by_user import (
    ROOT,
    load_training,
    score,
)


def user_weights(users, power):
    unique, counts = np.unique(users, return_counts=True)
    count_by_user = dict(zip(unique, counts))
    weights = np.asarray([
        count_by_user[user] ** (-power) for user in users
    ], dtype=np.float64)
    return weights / weights.mean()


def run(seed=10086):
    clean, noise, labels, users = load_training()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(splitter.split(clean, labels, groups=users))
    result = {
        "protocol": "geolife-60s-user-balanced-hgb-cv-v1",
        "seed": seed,
        "settings": {},
    }
    for power in (0.5, 1.0):
        folds = []
        for fold, (train_index, val_index) in enumerate(splits):
            model = HistGradientBoostingClassifier(
                learning_rate=0.07, max_iter=180, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0,
                class_weight="balanced", random_state=seed + fold,
            )
            train_x = np.concatenate([clean[train_index], noise[train_index]])
            train_y = np.concatenate([labels[train_index], labels[train_index]])
            weights = user_weights(users[train_index], power)
            model.fit(train_x, train_y,
                      sample_weight=np.concatenate([weights, weights]))
            generator = np.random.default_rng(seed + fold)
            val_x = clean[val_index].copy()
            use_noise = generator.random(len(val_index)) < 0.5
            val_x[use_noise] = noise[val_index][use_noise]
            folds.append({"fold": fold, **score(model, val_x, labels[val_index])})
            print(power, folds[-1], flush=True)
        result["settings"][str(power)] = {
            "folds": folds,
            "accuracy_mean": float(np.mean([x["accuracy"] for x in folds])),
            "accuracy_std": float(np.std([x["accuracy"] for x in folds])),
            "macro_f1_mean": float(np.mean([x["macro_f1"] for x in folds])),
            "macro_f1_std": float(np.std([x["macro_f1"] for x in folds])),
        }
    output = ROOT / "experiments/geolife_user_balanced_hgb_seed10086"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
