#!/usr/bin/env python3
"""Diagnose rate-dependent behavior masks in the five-rate GeoLife data.

The diagnostic is intentionally read-only with respect to generated feature
data.  A mask value of zero means that the observation is masked.  Feature
mask statistics are restricted to the motion channels consumed by B0.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ("Walk", "Bike", "Bus", "Car", "Train")
DEFAULT_MOTION_CHANNELS = (3, 4, 5, 8)


def _vector(value, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != expected_length:
        raise ValueError(
            f"{name} has length {vector.size}, expected {expected_length}"
        )
    return vector


def _masked(vector: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(vector)):
        raise ValueError("mask contains NaN or infinite values")
    return vector <= 0.5


def _finite_delta_t(value, *, expected_length: int, nominal_rate: int) -> np.ndarray:
    delta_t = _vector(value, expected_length=expected_length, name="delta_t")
    valid = np.isfinite(delta_t) & (delta_t > 0)
    replacement = float(np.median(delta_t[valid])) if np.any(valid) else float(nominal_rate)
    return np.where(valid, delta_t, replacement)


def summarize_samples(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot summarize an empty sample collection")

    def values(key: str) -> np.ndarray:
        return np.asarray([sample[key] for sample in samples], dtype=np.float64)

    return {
        "num_samples": len(samples),
        "avg_sequence_length": float(np.mean(values("sequence_length"))),
        "median_sequence_length": float(np.median(values("sequence_length"))),
        "avg_segment_duration": float(np.mean(values("segment_duration"))),
        "median_segment_duration": float(np.median(values("segment_duration"))),
        "avg_ts_masked_points": float(np.mean(values("ts_masked_points"))),
        "median_ts_masked_points": float(np.median(values("ts_masked_points"))),
        "avg_ts_mask_ratio": float(np.mean(values("ts_mask_ratio"))),
        "median_ts_mask_ratio": float(np.median(values("ts_mask_ratio"))),
        "avg_ts_mask_physical_duration": float(np.mean(values("ts_mask_duration"))),
        "median_ts_mask_physical_duration": float(np.median(values("ts_mask_duration"))),
        "avg_fs_mask_ratio": float(np.mean(values("fs_mask_ratio"))),
        "median_fs_mask_ratio": float(np.median(values("fs_mask_ratio"))),
        "avg_fs_mask_physical_duration": float(np.mean(values("fs_mask_duration"))),
        "median_fs_mask_physical_duration": float(np.median(values("fs_mask_duration"))),
        "fully_masked_ts_sample_ratio": float(np.mean(values("ts_fully_masked"))),
        "fully_masked_fs_channel_ratio": float(np.mean(values("fs_fully_masked_fraction"))),
    }


def analyze_rate(
    data_root: Path,
    rate: int,
    split: str,
    motion_channels: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    base = data_root / f"geolife_five_rate_fixed_{rate}s_features" / split
    required = {
        "trajectory": base / "noise_trj_segs.npy",
        "features": base / "noise_multi_feature_segs.npy",
        "labels": base / "noise_multi_feature_seg_labels.npy",
        "ts_masks": base / "trj_seg_masks.npy",
        "fs_masks": base / "fs_seg_masks.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing diagnostic inputs: " + ", ".join(missing))

    trajectories = np.load(required["trajectory"], allow_pickle=True)
    features = np.load(required["features"], allow_pickle=True)
    labels = np.asarray(np.load(required["labels"], allow_pickle=True)).reshape(-1)
    ts_masks = np.load(required["ts_masks"], allow_pickle=True)
    fs_masks = np.load(required["fs_masks"], allow_pickle=True)

    size = len(labels)
    lengths = {
        "trajectory": len(trajectories),
        "features": len(features),
        "ts_masks": len(ts_masks),
        "fs_masks": len(fs_masks),
    }
    if any(length != size for length in lengths.values()):
        raise ValueError(f"sample count mismatch: labels={size}, arrays={lengths}")
    if fs_masks.ndim != 2:
        raise ValueError(f"expected a 2-D feature-mask matrix, got {fs_masks.shape}")
    if max(motion_channels) >= fs_masks.shape[1]:
        raise ValueError(
            f"motion channel {max(motion_channels)} unavailable in {fs_masks.shape[1]} channels"
        )

    samples: list[dict] = []
    for index in range(size):
        trajectory_row = trajectories[index]
        sequence_length = len(np.asarray(trajectory_row[0]).reshape(-1))
        if sequence_length <= 0:
            raise ValueError(f"sample {index} has an empty trajectory")

        ts_mask = _masked(
            _vector(
                ts_masks[index],
                expected_length=sequence_length,
                name=f"TS mask sample {index}",
            )
        )
        delta_t = _finite_delta_t(
            features[index, 0],
            expected_length=sequence_length,
            nominal_rate=rate,
        )
        # S4 interpolates N-1 original intervals to N feature positions.
        # Scale back to interval count so a fully masked segment does not
        # exceed its approximate observed temporal span.
        interval_scale = max(sequence_length - 1, 0) / sequence_length
        segment_duration = float(np.sum(delta_t) * interval_scale)

        fs_ratios = []
        fs_durations = []
        fs_fully_masked = []
        for channel in motion_channels:
            channel_mask = _masked(
                _vector(
                    fs_masks[index, channel],
                    expected_length=sequence_length,
                    name=f"FS mask sample {index} channel {channel}",
                )
            )
            fs_ratios.append(float(np.mean(channel_mask)))
            fs_durations.append(float(np.sum(delta_t[channel_mask]) * interval_scale))
            fs_fully_masked.append(float(np.all(channel_mask)))

        label = int(labels[index])
        if not 0 <= label < len(CLASS_NAMES):
            raise ValueError(f"unsupported label {label} at sample {index}")
        samples.append(
            {
                "label": label,
                "sequence_length": sequence_length,
                "segment_duration": segment_duration,
                "ts_masked_points": int(np.sum(ts_mask)),
                "ts_mask_ratio": float(np.mean(ts_mask)),
                "ts_mask_duration": float(np.sum(delta_t[ts_mask]) * interval_scale),
                "ts_fully_masked": float(np.all(ts_mask)),
                "fs_mask_ratio": float(np.mean(fs_ratios)),
                "fs_mask_duration": float(np.mean(fs_durations)),
                "fs_fully_masked_fraction": float(np.mean(fs_fully_masked)),
            }
        )

    overall = summarize_samples(samples)
    overall.update({"rate": rate, "split": split, "data_dir": str(base)})
    by_class = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_samples = [sample for sample in samples if sample["label"] == class_id]
        if not class_samples:
            continue
        summary = summarize_samples(class_samples)
        summary.update({"rate": rate, "class_id": class_id, "class_name": class_name})
        by_class.append(summary)
    return overall, by_class


def evaluate_hypothesis(overall: list[dict], by_class: list[dict], window_seconds: float) -> dict:
    rate_map = {int(row["rate"]): row for row in overall}
    reasons = []
    criteria = {}

    if all(rate in rate_map for rate in (5, 10, 30, 60)):
        high_ts = np.mean([rate_map[rate]["avg_ts_mask_ratio"] for rate in (5, 10)])
        low_ts = np.mean([rate_map[rate]["avg_ts_mask_ratio"] for rate in (30, 60)])
        high_fs = np.mean([rate_map[rate]["avg_fs_mask_ratio"] for rate in (5, 10)])
        low_fs = np.mean([rate_map[rate]["avg_fs_mask_ratio"] for rate in (30, 60)])
        criteria["A"] = bool((low_ts - high_ts >= 0.05) or (low_fs - high_fs >= 0.05))
        if criteria["A"]:
            reasons.append(
                "30/60 s mean mask ratio exceeds 5/10 s by at least 5 percentage points"
            )
        criteria["A_details"] = {
            "high_rate_ts_mask_ratio": float(high_ts),
            "low_rate_ts_mask_ratio": float(low_ts),
            "high_rate_fs_mask_ratio": float(high_fs),
            "low_rate_fs_mask_ratio": float(low_fs),
        }
    else:
        criteria["A"] = False

    if 60 in rate_map:
        duration = rate_map[60]["avg_ts_mask_physical_duration"]
        criteria["B"] = bool(duration >= 0.5 * window_seconds)
        criteria["B_details"] = {
            "avg_60s_ts_mask_duration": duration,
            "window_seconds": window_seconds,
            "duration_fraction": duration / window_seconds,
        }
        if criteria["B"]:
            reasons.append("60 s mean TS masked duration covers at least half of the 300 s window")
    else:
        criteria["B"] = False

    criteria["C"] = None
    criteria["C_details"] = "EP counts, KDE duplicate mappings and fallback events are not recoverable from saved masks"

    class_map = {(int(row["rate"]), row["class_name"]): row for row in by_class}
    fragile_increases = []
    for class_name in ("Bus", "Train"):
        if all((rate, class_name) in class_map for rate in (5, 10, 30, 60)):
            high = np.mean(
                [class_map[(rate, class_name)]["avg_ts_mask_ratio"] for rate in (5, 10)]
            )
            low = np.mean(
                [class_map[(rate, class_name)]["avg_ts_mask_ratio"] for rate in (30, 60)]
            )
            fragile_increases.append({"class_name": class_name, "increase": float(low - high)})
    criteria["D"] = bool(any(item["increase"] >= 0.05 for item in fragile_increases))
    criteria["D_details"] = fragile_increases
    if criteria["D"]:
        reasons.append("Bus or Train TS mask ratio increases by at least 5 percentage points at 30/60 s")

    supported = bool(criteria["A"] or criteria["B"] or criteria["D"])
    return {
        "supported": supported,
        "verdict": "GO" if supported else "STOP",
        "criteria": criteria,
        "reasons": reasons,
    }


def _csv_rows(overall: list[dict], by_class: list[dict]) -> list[dict]:
    rows = []
    for record in overall:
        rows.append({"scope": "overall", "class_id": "", "class_name": "", **record})
    for record in by_class:
        rows.append({"scope": "class", **record})
    return rows


def write_csv(path: Path, overall: list[dict], by_class: list[dict]) -> None:
    rows = _csv_rows(overall, by_class)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and key != "data_dir":
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(
    path: Path,
    overall: list[dict],
    by_class: list[dict],
    hypothesis: dict,
    split: str,
    motion_channels: tuple[int, ...],
) -> None:
    lines = [
        "# GeoLife Behavior Mask degradation diagnosis v1",
        "",
        f"- Split: `{split}`",
        f"- Motion feature channels: `{list(motion_channels)}`",
        "- Mask convention: `0 = masked`, `1 = retained`",
        "- Physical duration: positive `delta_t` support corrected from N interpolated values to N-1 intervals",
        "",
        "## Five-rate summary",
        "",
        "| Rate | Samples | Avg points | Median points | Avg segment (s) | TS mask ratio | FS mask ratio | TS duration (s) | Fully masked TS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['rate']} s | {row['num_samples']} | "
            f"{row['avg_sequence_length']:.2f} | {row['median_sequence_length']:.2f} | "
            f"{row['avg_segment_duration']:.2f} | "
            f"{_percentage(row['avg_ts_mask_ratio'])} | "
            f"{_percentage(row['avg_fs_mask_ratio'])} | "
            f"{row['avg_ts_mask_physical_duration']:.2f} | "
            f"{_percentage(row['fully_masked_ts_sample_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-class summary",
            "",
            "| Rate | Class | Samples | Avg points | TS mask ratio | FS mask ratio | TS duration (s) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_class:
        lines.append(
            f"| {row['rate']} s | {row['class_name']} | {row['num_samples']} | "
            f"{row['avg_sequence_length']:.2f} | "
            f"{_percentage(row['avg_ts_mask_ratio'])} | "
            f"{_percentage(row['avg_fs_mask_ratio'])} | "
            f"{row['avg_ts_mask_physical_duration']:.2f} |"
        )

    lines.extend(["", "## Hypothesis decision", "", f"**{hypothesis['verdict']}**", ""])
    if hypothesis["reasons"]:
        lines.extend(f"- {reason}" for reason in hypothesis["reasons"])
    else:
        lines.append("- None of the pre-registered degradation criteria was met.")
    lines.extend(
        [
            "",
            "## Diagnostic limitations",
            "",
            "- Saved masks do not preserve KDE peak counts, duplicate mappings, EP counts, or fallback events.",
            "- Those events require optional instrumentation during future feature regeneration.",
            "- This diagnosis uses the training split only; the test split was not inspected for method design.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=int, nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--motion-channels",
        type=int,
        nargs="+",
        default=list(DEFAULT_MOTION_CHANNELS),
    )
    parser.add_argument("--window-seconds", type=float, default=300.0)
    args = parser.parse_args()

    rates = sorted(set(args.rates))
    motion_channels = tuple(args.motion_channels)
    overall = []
    by_class = []
    for rate in rates:
        rate_overall, rate_classes = analyze_rate(
            args.data_root, rate, args.split, motion_channels
        )
        overall.append(rate_overall)
        by_class.extend(rate_classes)

    hypothesis = evaluate_hypothesis(overall, by_class, args.window_seconds)
    payload = {
        "protocol": "geolife-behavior-mask-diagnosis-v1",
        "split": args.split,
        "rates": rates,
        "motion_channels": list(motion_channels),
        "window_seconds": args.window_seconds,
        "overall": overall,
        "by_class": by_class,
        "diagnostic_availability": {
            "ep_count": False,
            "no_ep_ratio": False,
            "fallback_ratio": False,
            "kde_duplicate_mapping": False,
        },
        "hypothesis": hypothesis,
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    write_csv(csv_path, overall, by_class)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(
        markdown_path,
        overall,
        by_class,
        hypothesis,
        args.split,
        motion_channels,
    )
    print(json.dumps({
        "verdict": hypothesis["verdict"],
        "reasons": hypothesis["reasons"],
        "outputs": [str(csv_path), str(json_path), str(markdown_path)],
    }, indent=2))


if __name__ == "__main__":
    main()
