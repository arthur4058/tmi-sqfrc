"""Verify the frozen low-rate experiment against a paired B0 baseline.

This command does not fit, calibrate, or select a model.  It only checks a
completed test artifact against a predeclared dual-metric threshold.  Both
metrics must come from the baseline and candidate stored in the same result
file so that results from different seeds, splits, or checkpoints cannot be
mixed accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("accuracy", "macro_f1")
TOLERANCE = 1e-12


def verify_acceptance(
        result: dict,
        minimum_delta: float = 0.01,
        expected_rate: int = 60,
        expected_samples: int | None = None) -> dict:
    """Return a strict paired acceptance decision for a frozen test result."""
    if minimum_delta < 0:
        raise ValueError("minimum delta must be non-negative")
    if result.get("test_fit_performed") is not False:
        raise ValueError("test result must explicitly record no test-time fit")
    if result.get("sampling_interval_seconds") != expected_rate:
        raise ValueError("sampling interval does not match the target protocol")
    if expected_samples is not None and result.get("test_samples") != expected_samples:
        raise ValueError("test sample count does not match the frozen split")

    baseline = result.get("baseline")
    candidate = result.get("class_aware_stacker")
    stored_delta = result.get("delta_vs_baseline")
    if not all(isinstance(item, dict) for item in (
            baseline, candidate, stored_delta)):
        raise ValueError("paired baseline, candidate, and deltas are required")

    computed_delta = {}
    metric_passed = {}
    for metric in METRICS:
        try:
            delta = float(candidate[metric]) - float(baseline[metric])
            recorded = float(stored_delta[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {metric} values") from error
        if abs(delta - recorded) > TOLERANCE:
            raise ValueError(f"stored {metric} delta is not paired consistently")
        computed_delta[metric] = delta
        metric_passed[metric] = delta + TOLERANCE >= minimum_delta

    return {
        "protocol": result.get("protocol"),
        "sampling_interval_seconds": expected_rate,
        "test_samples": result.get("test_samples"),
        "minimum_delta": minimum_delta,
        "baseline": {metric: float(baseline[metric]) for metric in METRICS},
        "candidate": {metric: float(candidate[metric]) for metric in METRICS},
        "computed_delta": computed_delta,
        "metric_passed": metric_passed,
        "accepted": all(metric_passed.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify paired Accuracy and Macro-F1 improvements."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--minimum-delta", type=float, default=0.01)
    parser.add_argument("--rate", type=int, default=60)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    decision = verify_acceptance(
        result,
        minimum_delta=args.minimum_delta,
        expected_rate=args.rate,
        expected_samples=args.expected_samples,
    )
    rendered = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if decision["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
