"""Audit the completed representative-method comparison artifact."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

METHODS = (
    "takahashi_rf",
    "xgboost",
    "dabiri_cnn",
    "deepinsight_vit",
    "maso_msf",
)
RATES = (5, 10, 20, 30, 60)
CLASSES = ("Walk", "Bike", "Bus", "Car", "Train")


def main(path: str) -> None:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol = artifact["protocol"]
    protocol_sha = protocol["protocol_sha256"]
    results = artifact["results"]

    expected = {(method, rate) for method in METHODS for rate in RATES}
    observed = {(row["method"], int(row["rate_seconds"])) for row in results}
    if len(results) != len(expected) or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AssertionError(
            f"expected {len(expected)} unique cells, got {len(results)} rows; "
            f"missing={missing}, extra={extra}"
        )

    fingerprints: dict[int, set[str]] = defaultdict(set)
    for row in results:
        if row["seed"] != 10086:
            raise AssertionError(f"unexpected seed: {row}")
        if row["protocol_sha256"] != protocol_sha:
            raise AssertionError(f"protocol SHA mismatch: {row}")
        rate = int(row["rate_seconds"])
        fingerprints[rate].add(row["test_fingerprint"])
        values = [row["accuracy"], row["macro_f1"], *row["per_class_f1"].values()]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise AssertionError(f"invalid metric: {row}")
        if set(row["per_class_f1"]) != set(CLASSES):
            raise AssertionError(f"class mismatch: {row}")
        macro = sum(row["per_class_f1"].values()) / len(CLASSES)
        if not math.isclose(macro, row["macro_f1"], rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"macro-F1 mismatch: {row}")

    bad_rates = {rate: values for rate, values in fingerprints.items() if len(values) != 1}
    if bad_rates:
        raise AssertionError(f"methods used different test labels/users: {bad_rates}")

    users = protocol["user_partitions"]
    split_sets = {name: set(values) for name, values in users.items()}
    if tuple(len(split_sets[name]) for name in ("train", "val", "test")) != (44, 6, 12):
        raise AssertionError(f"unexpected user counts: {users}")
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("user split is not disjoint")

    print(f"validated_cells={len(results)}")
    print(f"protocol_sha256={protocol_sha}")
    print("user_split=44/6/12 disjoint")
    print("test_fingerprint=identical across methods for every rate")
    print("metrics=finite and Macro-F1 recomputation passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_JSON")
    main(sys.argv[1])
