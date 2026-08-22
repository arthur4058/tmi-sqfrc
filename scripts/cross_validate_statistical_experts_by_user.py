"""Three-fold user-disjoint audit for 60-second statistical experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.train_statistical_motion_expert import DATA_ROOT, summarize_arrays


ROOT = Path(__file__).resolve().parents[1]


def load_training():
    base = DATA_ROOT / "train"
    clean = summarize_arrays(
        np.load(base / "clean_multi_feature_segs.npy", allow_pickle=True),
        np.load(base / "clean_trj_segs.npy", allow_pickle=True),
    )
    noise = summarize_arrays(
        np.load(base / "noise_multi_feature_segs.npy", allow_pickle=True),
        np.load(base / "noise_trj_segs.npy", allow_pickle=True),
    )
    labels = np.load(base / "clean_multi_feature_seg_labels.npy").astype(np.int64)
    users = np.load(base / "segment_user_ids.npy").astype(np.int64)
    return clean, noise, labels, users


def score(model, values, labels):
    prediction = model.predict(values)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


def factories(seed):
    return {
        "extra_trees": lambda fold: ExtraTreesClassifier(
            n_estimators=220, max_features=0.7, min_samples_leaf=3,
            max_depth=22, class_weight="balanced", n_jobs=-1,
            random_state=seed + fold,
        ),
        "hist_gradient_boosting": lambda fold: HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=180, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            class_weight="balanced", random_state=seed + fold,
        ),
    }


def run(seed=10086):
    clean, noise, labels, users = load_training()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    result = {
        "protocol": "geolife-60s-training-user-cv-v1",
        "seed": seed,
        "samples": int(len(labels)),
        "users": int(len(np.unique(users))),
        "models": {},
    }
    splits = list(splitter.split(clean, labels, groups=users))
    for name, factory in factories(seed).items():
        folds = []
        for fold, (train_index, val_index) in enumerate(splits):
            model = factory(fold)
            train_x = np.concatenate([clean[train_index], noise[train_index]])
            train_y = np.concatenate([labels[train_index], labels[train_index]])
            model.fit(train_x, train_y)
            # Deterministic 50/50 clean/noise view, matching the main protocol.
            generator = np.random.default_rng(seed + fold)
            val_x = clean[val_index].copy()
            use_noise = generator.random(len(val_index)) < 0.5
            val_x[use_noise] = noise[val_index][use_noise]
            fold_score = score(model, val_x, labels[val_index])
            folds.append({
                "fold": fold,
                "train_users": int(len(np.unique(users[train_index]))),
                "val_users": int(len(np.unique(users[val_index]))),
                "val_samples": int(len(val_index)),
                **fold_score,
            })
            print(name, folds[-1], flush=True)
        result["models"][name] = {
            "folds": folds,
            "accuracy_mean": float(np.mean([item["accuracy"] for item in folds])),
            "accuracy_std": float(np.std([item["accuracy"] for item in folds])),
            "macro_f1_mean": float(np.mean([item["macro_f1"] for item in folds])),
            "macro_f1_std": float(np.std([item["macro_f1"] for item in folds])),
        }
    output = ROOT / "experiments/geolife_statistical_user_cv_seed10086"
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
