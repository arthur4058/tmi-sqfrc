"""Train representative methods under one user-disjoint five-rate protocol."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.fair_comparison_models import (
    DabiriCNN, DeepInsightViT, FeatureImageMapper, MASOMSFAdapter)
from scripts.fair_comparison_protocol import (
    CLASS_NAMES, RATES, ROOT, evaluate_probabilities, fit_channel_normalizer,
    load_split, normalize_values, seed_everything, summarize, validate_protocol)

METHODS = ("takahashi_rf", "xgboost", "dabiri_cnn", "deepinsight_vit",
           "maso_msf")


def probabilities_from_logits(logits):
    shifted = logits - logits.max(1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(1, keepdims=True)


def train_takahashi(rate, seed, output):
    train, val, test = (load_split(rate, split) for split in ("train", "val", "test"))
    x_train, x_test = summarize(train), summarize(test)
    model = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    model.fit(x_train, train.labels)
    probabilities = model.predict_proba(x_test)
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    return probabilities, test


def train_xgboost(rate, seed, output):
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("install pinned xgboost before running this method") from error
    train, val, test = (load_split(rate, split) for split in ("train", "val", "test"))
    scaler = StandardScaler().fit(summarize(train))
    x_train = scaler.transform(summarize(train))
    x_val = scaler.transform(summarize(val))
    x_test = scaler.transform(summarize(test))
    model = XGBClassifier(
        n_estimators=800, max_depth=7, learning_rate=.04, subsample=.85,
        colsample_bytree=.85, objective="multi:softprob", num_class=5,
        eval_metric="mlogloss", tree_method="hist", random_state=seed,
        n_jobs=12, early_stopping_rounds=40)
    model.fit(x_train, train.labels, eval_set=[(x_val, val.labels)], verbose=False)
    probabilities = model.predict_proba(x_test)
    model.save_model(output / "model.json")
    with (output / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    return probabilities, test


def neural_arrays(method, rate):
    train, val, test = (load_split(rate, split) for split in ("train", "val", "test"))
    if method == "deepinsight_vit":
        summaries = tuple(summarize(data) for data in (train, val, test))
        mapper = FeatureImageMapper().fit(summaries[0])
        arrays = tuple(mapper.transform(values) for values in summaries)
        return arrays, (train, val, test), mapper
    mean, std = fit_channel_normalizer(train)
    width = max(int(np.asarray(data.mask).sum(1).max())
                for data in (train, val, test))
    arrays = tuple(normalize_values(data, mean, std)[:, :, :width]
                   for data in (train, val, test))
    return arrays, (train, val, test), {"mean": mean, "std": std, "width": width}


def make_model(method, arrays):
    if method == "dabiri_cnn":
        return DabiriCNN()
    if method == "maso_msf":
        return MASOMSFAdapter()
    if method == "deepinsight_vit":
        return DeepInsightViT(arrays[0].shape[-1])
    raise ValueError(method)


def predict(model, values, device, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(values)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    output = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            output.append(model(batch.to(device)).cpu().numpy())
    return probabilities_from_logits(np.concatenate(output))


def train_neural(method, rate, seed, output, epochs, patience, batch_size, device):
    arrays, splits, preprocessing = neural_arrays(method, rate)
    train, val, test = splits
    model = make_model(method, arrays).to(device)
    counts = np.bincount(train.labels, minlength=5)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays[0]), torch.from_numpy(train.labels)),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
        pin_memory=device.startswith("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, stale = -1.0, 0
    checkpoint = output / "model_best.pth"
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(values), labels, weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_probabilities(
            predict(model, arrays[1], device, batch_size), val.labels)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        **validation})
        print(f"{method} {rate}s epoch={epoch} loss={np.mean(losses):.5f} "
              f"val_acc={validation['accuracy']:.4f} "
              f"val_f1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best + 1e-6:
            best, stale = validation["macro_f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (output / "preprocessing.pkl").open("wb") as handle:
        pickle.dump(preprocessing, handle)
    return predict(model, arrays[2], device, batch_size), test


def run_one(method, rate, args, protocol):
    output = ROOT / "experiments/fair_representative_comparison" / method / f"{rate}s_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "result.json"
    if args.resume and result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    start = time.time()
    if method == "takahashi_rf":
        probabilities, test = train_takahashi(rate, args.seed, output)
    elif method == "xgboost":
        probabilities, test = train_xgboost(rate, args.seed, output)
    else:
        probabilities, test = train_neural(
            method, rate, args.seed, output, args.epochs, args.patience,
            args.batch_size, args.device)
    metrics = evaluate_probabilities(probabilities, test.labels)
    np.save(output / "test_probabilities.npy", probabilities.astype(np.float32))
    np.save(output / "test_labels.npy", test.labels)
    result = {
        "method": method, "rate_seconds": rate, "seed": args.seed,
        "test_fingerprint": test.fingerprint,
        "protocol_sha256": protocol["protocol_sha256"],
        "runtime_seconds": time.time() - start, **metrics,
    }
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def write_reports(results, protocol, seed):
    report_dir = ROOT / "reports/experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = report_dir / f"fair_representative_comparison_seed{seed}"
    (stem.with_suffix(".json")).write_text(
        json.dumps({"protocol": protocol, "results": results}, indent=2),
        encoding="utf-8")
    fields = ("method", "rate_seconds", "seed", "accuracy", "macro_f1",
              "runtime_seconds", "protocol_sha256", "test_fingerprint")
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rates", nargs="+", type=int, choices=RATES, default=list(RATES))
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_protocol(args.rebuild_cache)
    manifest = ROOT / "reports/manifests/fair_comparison_protocol.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"protocol_sha256={protocol['protocol_sha256']}", flush=True)
    if args.validate_only:
        return
    results = [run_one(method, rate, args, protocol)
               for method in args.methods for rate in args.rates]
    fingerprints = {}
    for row in results:
        key = row["rate_seconds"]
        fingerprints.setdefault(key, row["test_fingerprint"])
        if fingerprints[key] != row["test_fingerprint"]:
            raise RuntimeError(f"methods evaluated different test order at {key}s")
    write_reports(results, protocol, args.seed)


if __name__ == "__main__":
    main()
