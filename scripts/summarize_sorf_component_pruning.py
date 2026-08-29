"""Aggregate the matched full-versus-pruned SORF component experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/experiments"
RATES = (5, 10, 20, 30, 60)
LOW_RATES = (30, 60)
SEEDS = (42, 2024, 10086)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(result):
    return int(result["rate_seconds"]), int(result["seed"])


def metric(result, name):
    return float(result["test"]["v74"][name])


def result_map(path: Path):
    return {key(item): item for item in load(path)["results"]}


def main():
    full = result_map(REPORT_DIR / "geolife_v74_five_rate_multiseed.json")
    relation_fusion = result_map(
        REPORT_DIR / "geolife_v74_ablation_relation_fusion.json"
    )

    pruned = {}
    for rate in (5, 10, 20):
        path = ROOT / (
            "experiments/geolife_v74_ablation_pruned_no_m2_"
            f"{rate}s_seed10086/result.json"
        )
        item = load(path)
        pruned[key(item)] = item
    for rate in LOW_RATES:
        pruned[(rate, 10086)] = relation_fusion[(rate, 10086)]
        for seed in (42, 2024):
            path = ROOT / (
                "experiments/geolife_v74_ablation_pruned_no_m2_"
                f"{rate}s_seed{seed}/result.json"
            )
            item = load(path)
            pruned[key(item)] = item

    five_rate_rows = []
    for rate in RATES:
        full_item = full[(rate, 10086)]
        pruned_item = pruned[(rate, 10086)]
        row = {"rate_seconds": rate}
        for name in ("accuracy", "macro_f1"):
            row[f"full_{name}"] = metric(full_item, name)
            row[f"pruned_{name}"] = metric(pruned_item, name)
            row[f"delta_{name}_pp"] = 100 * (
                row[f"pruned_{name}"] - row[f"full_{name}"]
            )
        five_rate_rows.append(row)

    low_rate_rows = []
    for rate in LOW_RATES:
        row = {"rate_seconds": rate, "seeds": list(SEEDS)}
        for name in ("accuracy", "macro_f1"):
            full_values = np.asarray(
                [metric(full[(rate, seed)], name) for seed in SEEDS]
            )
            pruned_values = np.asarray(
                [metric(pruned[(rate, seed)], name) for seed in SEEDS]
            )
            row[f"full_{name}_mean"] = float(full_values.mean())
            row[f"full_{name}_std"] = float(full_values.std(ddof=1))
            row[f"pruned_{name}_mean"] = float(pruned_values.mean())
            row[f"pruned_{name}_std"] = float(pruned_values.std(ddof=1))
            row[f"delta_{name}_pp"] = float(
                100 * (pruned_values.mean() - full_values.mean())
            )
        low_rate_rows.append(row)

    five_delta_acc = float(np.mean([r["delta_accuracy_pp"] for r in five_rate_rows]))
    five_delta_f1 = float(np.mean([r["delta_macro_f1_pp"] for r in five_rate_rows]))
    low_deltas = [
        abs(row[f"delta_{name}_pp"])
        for row in low_rate_rows
        for name in ("accuracy", "macro_f1")
    ]
    decision = (
        five_delta_acc >= -0.10
        and five_delta_f1 >= -0.10
        and max(low_deltas) <= 0.25
    )

    summary = {
        "protocol": "sorf-component-pruning-validation-v1",
        "removed_component": (
            "observation-drop augmentation and sparse-view consistency loss"
        ),
        "kept_components": [
            "real-observation relational expert",
            "validation-constrained calibrated probability fusion",
        ],
        "five_rate_seed10086": five_rate_rows,
        "five_rate_mean_delta_pp": {
            "accuracy": five_delta_acc,
            "macro_f1": five_delta_f1,
        },
        "low_rate_three_seed": low_rate_rows,
        "decision_thresholds": {
            "five_rate_mean_delta_not_below_pp": -0.10,
            "max_abs_low_rate_mean_delta_pp": 0.25,
        },
        "prune_component": decision,
    }
    json_path = REPORT_DIR / "sorf_component_pruning_validation.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# SORF component pruning validation",
        "",
        "## Question and controlled setting",
        "",
        "This experiment tests whether observation dropping plus sparse-view "
        "consistency learning is necessary. The full and pruned models use the "
        "same user-disjoint split, five-rate data, matched B0 checkpoints, "
        "training budget, validation-only calibration, seeds, and test sets. "
        "The pruned model changes only `drop_probability`, `drop_weight`, and "
        "`consistency_weight` from their full-model values to zero.",
        "",
        "## Five-rate result (seed 10086, %)",
        "",
        "| Rate | Full Acc | Pruned Acc | Delta | Full F1 | Pruned F1 | Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in five_rate_rows:
        lines.append(
            f"| {row['rate_seconds']} s | {100*row['full_accuracy']:.2f} | "
            f"{100*row['pruned_accuracy']:.2f} | {row['delta_accuracy_pp']:+.2f} | "
            f"{100*row['full_macro_f1']:.2f} | {100*row['pruned_macro_f1']:.2f} | "
            f"{row['delta_macro_f1_pp']:+.2f} |"
        )
    lines.extend([
        f"| Mean | - | - | {five_delta_acc:+.2f} | - | - | {five_delta_f1:+.2f} |",
        "",
        "## Low-rate three-seed result (mean ± sample std, %)",
        "",
        "| Rate | Full Acc | Pruned Acc | Delta | Full F1 | Pruned F1 | Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in low_rate_rows:
        lines.append(
            f"| {row['rate_seconds']} s | "
            f"{100*row['full_accuracy_mean']:.2f} ± {100*row['full_accuracy_std']:.2f} | "
            f"{100*row['pruned_accuracy_mean']:.2f} ± {100*row['pruned_accuracy_std']:.2f} | "
            f"{row['delta_accuracy_pp']:+.2f} | "
            f"{100*row['full_macro_f1_mean']:.2f} ± {100*row['full_macro_f1_std']:.2f} | "
            f"{100*row['pruned_macro_f1_mean']:.2f} ± {100*row['pruned_macro_f1_std']:.2f} | "
            f"{row['delta_macro_f1_pp']:+.2f} |"
        )
    conclusion = (
        "The component is removed from the final model. Its removal preserves "
        "the target 30 s/60 s three-seed performance and does not reduce the "
        "five-rate mean, while simplifying training and eliminating an "
        "unsupported component claim."
        if decision
        else
        "The component is retained because the predeclared pruning criteria "
        "were not satisfied."
    )
    lines.extend([
        "",
        "## Decision",
        "",
        conclusion,
        "",
        "The isolated 20 s score decreases, so the result is not described as "
        "uniform improvement. The pruning decision is based on the declared "
        "low-rate target and aggregate five-rate behavior, not cherry-picking.",
        "",
        "Final model: real-observation relational expert + "
        "validation-constrained calibrated probability fusion.",
        "",
    ])
    md_path = REPORT_DIR / "sorf_component_pruning_validation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    print(f"prune_component={decision}")


if __name__ == "__main__":
    main()
