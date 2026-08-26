"""Generate audited 30/60-second B0 and SORF-TMI confusion matrices."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_dual_expert_consensus import softmax
from scripts.run_v74_five_rate_multiseed import (
    CLASS_NAMES,
    ROOT,
    RateGeneralizedRelationNet,
    anchor_count,
    b0_logits,
    load_split,
    metric_summary,
    relation_logits,
    select_aligned,
)


def load_result(rate: int, seed: int) -> dict:
    report = json.loads(
        (ROOT / "reports/experiments/geolife_v74_five_rate_multiseed.json")
        .read_text(encoding="utf-8")
    )
    matches = [
        row for row in report["results"]
        if row["rate_seconds"] == rate and row["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one stored result for rate={rate}, seed={seed}")
    return matches[0]


def expert_logits(rate: int, seed: int, device: str):
    checkpoint_path = (
        ROOT / f"experiments/geolife_v74_generalized_{rate}s_seed{seed}"
        / "model_best.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model = RateGeneralizedRelationNet(
        anchor_count(rate),
        width=checkpoint["settings"]["width"],
        layers=checkpoint["settings"]["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    values, mask, labels, _ = select_aligned(
        load_split(rate, "test"), seed,
        checkpoint["mean"], checkpoint["std"])
    logits = relation_logits(model, values, mask, device)
    return logits, labels


def audit_metrics(name: str, current: dict, expected: dict) -> None:
    for key in ("accuracy", "macro_f1"):
        if not np.isclose(current[key], expected[key], atol=1e-12, rtol=0.0):
            raise AssertionError(
                f"{name} {key} mismatch: {current[key]} != {expected[key]}"
            )


def evaluate_rate(rate: int, seed: int, device: str) -> dict:
    stored = load_result(rate, seed)
    base_logits, base_labels = b0_logits(rate, seed, "test", device)
    relation_output, relation_labels = expert_logits(rate, seed, device)
    if not np.array_equal(base_labels, relation_labels):
        raise AssertionError(f"B0/relation labels differ for {rate}s")

    b0_probability = softmax(base_logits, stored["temperatures"]["b0"])
    expert_probability = softmax(
        relation_output, stored["temperatures"]["relation_expert"])
    weight = stored["validation"]["fusion"]["weight"]
    sorf_probability = (
        (1.0 - weight) * b0_probability + weight * expert_probability
    )
    predictions = {
        "b0": b0_probability.argmax(1),
        "sorf_tmi": sorf_probability.argmax(1),
    }
    metrics = {
        "b0": metric_summary(b0_probability, base_labels),
        "sorf_tmi": metric_summary(sorf_probability, base_labels),
    }
    audit_metrics(f"{rate}s B0", metrics["b0"], stored["test"]["b0"])
    audit_metrics(
        f"{rate}s SORF-TMI", metrics["sorf_tmi"], stored["test"]["v74"])

    matrices = {}
    for name, prediction in predictions.items():
        counts = confusion_matrix(
            base_labels, prediction, labels=np.arange(len(CLASS_NAMES)))
        row_total = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts, row_total, out=np.zeros_like(counts, dtype=np.float64),
            where=row_total != 0)
        matrices[name] = {
            "counts": counts.tolist(),
            "row_normalized": normalized.tolist(),
        }
    b0_normalized = np.asarray(matrices["b0"]["row_normalized"])
    sorf_normalized = np.asarray(matrices["sorf_tmi"]["row_normalized"])
    delta = sorf_normalized - b0_normalized
    return {
        "rate_seconds": rate,
        "seed": seed,
        "samples": int(len(base_labels)),
        "fusion_weight": float(weight),
        "metrics": metrics,
        "matrices": matrices,
        "row_normalized_delta": delta.tolist(),
        "class_recall_delta": {
            name: float(delta[index, index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def annotate_matrix(axis, values, difference=False):
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column] * 100.0
            if difference:
                label = f"{value:+.1f}"
                scale = max(float(np.max(np.abs(values * 100.0))), 1e-12)
                color = "white" if abs(value) >= 0.55 * scale else "black"
            else:
                label = f"{value:.1f}"
                color = "white" if value >= 55.0 else "black"
            axis.text(
                column, row, label, ha="center", va="center",
                fontsize=8, color=color)


def save_svg(figure, output: Path) -> None:
    """Save a deterministic SVG without Git-hostile trailing whitespace."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight")
    svg = output.read_text(encoding="utf-8")
    output.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def plot_matrices(results: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)
    last_image = None
    for row, result in enumerate(results):
        for column, key in enumerate(("b0", "sorf_tmi")):
            axis = axes[row, column]
            values = np.asarray(result["matrices"][key]["row_normalized"])
            last_image = axis.imshow(values * 100.0, vmin=0.0, vmax=100.0, cmap="Blues")
            annotate_matrix(axis, values)
            axis.set_title(
                f"{result['rate_seconds']} s — "
                f"{'B0' if key == 'b0' else 'SORF-TMI'}",
                fontsize=12)
            axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
            axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
    figure.colorbar(
        last_image, ax=axes, shrink=0.82, label="Row-normalized proportion (%)")
    figure.suptitle(
        "B0 and SORF-TMI confusion matrices at low sampling rates",
        fontsize=14)
    save_svg(figure, output)
    plt.close(figure)


def plot_deltas(results: list[dict], output: Path) -> None:
    matrices = [np.asarray(result["row_normalized_delta"]) * 100.0 for result in results]
    limit = max(1.0, max(float(np.abs(matrix).max()) for matrix in matrices))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    last_image = None
    for axis, result, values in zip(axes, results, matrices):
        last_image = axis.imshow(values, vmin=-limit, vmax=limit, cmap="RdBu")
        annotate_matrix(axis, values / 100.0, difference=True)
        axis.set_title(f"{result['rate_seconds']} s — SORF-TMI minus B0", fontsize=12)
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    figure.colorbar(
        last_image, ax=axes, shrink=0.82,
        label="Change in row-normalized proportion (percentage points)")
    figure.suptitle("Confusion-pattern changes after relational fusion", fontsize=14)
    save_svg(figure, output)
    plt.close(figure)


def markdown_report(payload: dict) -> str:
    lines = [
        "# SORF-TMI low-rate confusion-matrix audit",
        "",
        f"> Seed: {payload['seed']}",
        "> Checkpoints, temperatures and fusion weights are identical to the "
        "audited five-rate main experiment.",
        "",
    ]
    for result in payload["results"]:
        rate = result["rate_seconds"]
        lines.extend((
            f"## {rate} seconds",
            "",
            "| Model | Accuracy | Macro-F1 | " + " | ".join(CLASS_NAMES) + " |",
            "|---|---:|---:|" + "---:|" * len(CLASS_NAMES),
        ))
        for key, label in (("b0", "B0"), ("sorf_tmi", "SORF-TMI")):
            metrics = result["metrics"][key]
            class_values = [100.0 * metrics["per_class_f1"][name] for name in CLASS_NAMES]
            lines.append(
                f"| {label} | {100 * metrics['accuracy']:.2f} | "
                f"{100 * metrics['macro_f1']:.2f} | "
                + " | ".join(f"{value:.2f}" for value in class_values) + " |"
            )
        lines.extend((
            "",
            "Class recall changes (SORF-TMI − B0): "
            + ", ".join(
                f"{name} {100 * value:+.2f} pp"
                for name, value in result["class_recall_delta"].items()
            )
            + ".",
            "",
        ))
    lines.extend((
        "## Audit status",
        "",
        "- Recomputed B0 Accuracy/Macro-F1 exactly match the stored main result.",
        "- Recomputed SORF-TMI Accuracy/Macro-F1 exactly match the stored main result.",
        "- B0 and relation-expert labels are identical for each rate.",
        "- Raw counts and row-normalized matrices are stored in the JSON artifact.",
        "",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=int, default=[30, 60])
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="reports/experiments/sorf_tmi_confusion_30s_60s_seed10086.json",
    )
    parser.add_argument(
        "--matrix-figure", default="docs/figures/fig5_confusion_matrices.svg")
    parser.add_argument(
        "--delta-figure", default="docs/figures/fig6_confusion_delta.svg")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    results = [evaluate_rate(rate, args.seed, args.device) for rate in args.rates]
    payload = {
        "protocol": "sorf-tmi-confusion-audit-v1",
        "seed": args.seed,
        "class_names": list(CLASS_NAMES),
        "normalization": "rows sum to one; rows=true, columns=predicted",
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    output.with_suffix(".md").write_text(
        markdown_report(payload), encoding="utf-8")
    plot_matrices(results, ROOT / args.matrix_figure)
    plot_deltas(results, ROOT / args.delta_figure)
    print(output)
    print(output.with_suffix(".md"))
    print(ROOT / args.matrix_figure)
    print(ROOT / args.delta_figure)


if __name__ == "__main__":
    main()
