"""Combine controlled external baselines with existing B0/SORF results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATES = (5, 10, 20, 30, 60)
DISPLAY = {
    "takahashi_rf": "Takahashi GPS-only RF (2026)",
    "xgboost": "XGBoost",
    "dabiri_cnn": "Dabiri CNN (2018)",
    "deepinsight_vit": "DeepInsight-ViT (2025, adapted)",
    "maso_msf": "MASO-MSF (2023, adapted)",
    "b0": "B0 dual-branch Transformer",
    "sorf_tmi": "SORF-TMI",
}


def load_rows(seed: int):
    external_path = ROOT / f"reports/experiments/fair_representative_comparison_seed{seed}.json"
    v74_path = ROOT / "reports/experiments/geolife_v74_five_rate_multiseed.json"
    external = json.loads(external_path.read_text(encoding="utf-8"))
    rows = list(external["results"])
    v74 = json.loads(v74_path.read_text(encoding="utf-8"))["results"]
    for item in v74:
        if item["seed"] != seed or item["rate_seconds"] not in RATES:
            continue
        for method, key in (("b0", "b0"), ("sorf_tmi", "v74")):
            metrics = item["test"][key]
            rows.append({
                "method": method,
                "rate_seconds": item["rate_seconds"],
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            })
    return external["protocol"], rows


def main():
    seed = 10086
    protocol, rows = load_rows(seed)
    by_key = {(row["method"], row["rate_seconds"]): row for row in rows}
    methods = tuple(DISPLAY)
    missing = [(method, rate) for method in methods for rate in RATES
               if (method, rate) not in by_key]
    if missing:
        raise RuntimeError(f"incomplete comparison cells: {missing}")
    output = ROOT / f"reports/experiments/fair_representative_comparison_seed{seed}.md"
    lines = [
        "# Representative-method comparison under the shared five-rate protocol",
        "",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`",
        "",
        "All values below are newly evaluated on the same user-disjoint test sets. "
        "They are not copied from the cited papers. Each cell is Accuracy/Macro-F1 (%).",
        "",
        "| Method | 5 s | 10 s | 20 s | 30 s | 60 s | Mean Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for method in methods:
        cells, f1_values = [], []
        for rate in RATES:
            row = by_key[(method, rate)]
            cells.append(f"{100 * row['accuracy']:.2f}/{100 * row['macro_f1']:.2f}")
            f1_values.append(row["macro_f1"])
            csv_rows.append({"method": method, "rate_seconds": rate,
                             "accuracy": row["accuracy"],
                             "macro_f1": row["macro_f1"], "seed": seed})
        lines.append(f"| {DISPLAY[method]} | " + " | ".join(cells)
                     + f" | {100 * sum(f1_values) / len(f1_values):.2f} |")
    lines.extend((
        "", "## Interpretation boundary", "",
        "The MASO-MSF and DeepInsight-ViT rows are controlled architecture/input "
        "adaptations because their original data products and random splits are "
        "incompatible with the shared protocol. Takahashi is GPS-only: GIS/POI "
        "features are excluded. These labels must be retained in the paper.", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    csv_path = output.with_suffix(".table.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, csv_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output)


if __name__ == "__main__":
    main()
