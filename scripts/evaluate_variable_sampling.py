#!/usr/bin/env python3
"""Evaluate one fixed-5s checkpoint on all variable-sampling test views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


CONDITIONS = {
    "fixed_5s": "geolife_user_fixed_5s",
    "fixed_10s": "geolife_user_fixed_10s",
    "fixed_20s": "geolife_user_fixed_20s",
    "fixed_30s": "geolife_user_fixed_30s",
    "fixed_60s": "geolife_user_fixed_60s",
    "random_drop_30": "geolife_user_random_drop_30",
    "random_drop_50": "geolife_user_random_drop_50",
    "random_drop_70": "geolife_user_random_drop_70",
    "continuous_gap_30": "geolife_user_continuous_gap_30",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def find_average_row(report: pd.DataFrame) -> pd.Series:
    if "class" in report.columns:
        labels = report["class"].astype(str).str.lower()
        average_rows = report[
            labels.str.contains("avg") | labels.str.contains("total")
        ]
        if not average_rows.empty:
            return average_rows.iloc[-1]
    for index in reversed(report.index.tolist()):
        if "avg" in str(index).lower() or "total" in str(index).lower():
            return report.loc[index]
    return report.iloc[-1]


def parse_result(workbook: Path) -> dict:
    report = pd.read_excel(
        workbook, sheet_name="Classification Report", index_col=0
    )
    confusion = pd.read_excel(
        workbook, sheet_name="Confusion Matrix", index_col=0
    )
    average = find_average_row(report)

    f1_column = "f1-score" if "f1-score" in report.columns else "f1"
    frequency_column = next(
        (
            column
            for column in ("abs. freq.", "support", "frequency")
            if column in report.columns
        ),
        None,
    )
    per_class = {}
    for index in report.index:
        row = report.loc[index]
        class_name = str(row.get("class", index))
        if "avg" in class_name.lower() or "total" in class_name.lower():
            continue
        per_class[class_name] = {
            "precision": native(row.get("precision")),
            "recall": native(row.get("recall")),
            "f1": native(row.get(f1_column)),
            "samples": native(row.get(frequency_column))
            if frequency_column
            else None,
        }

    return {
        "accuracy": native(average.get("accuracy")),
        "macro_precision": native(average.get("precision")),
        "macro_recall": native(average.get("recall")),
        "macro_f1": native(average.get(f1_column)),
        "samples": native(average.get(frequency_column))
        if frequency_column
        else None,
        "per_class": per_class,
        "confusion_matrix": {
            str(row): {
                str(column): native(confusion.loc[row, column])
                for column in confusion.columns
            }
            for row in confusion.index
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(
            "configs/variable_sampling/geolife_user_fixed5_seed10086.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/geolife_user_fixed5_seed10086/"
            "checkpoints/model_best.pth"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/geolife_variable_sampling_tests"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/experiments"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_config_path = (repo / args.base_config).resolve()
    checkpoint = (repo / args.checkpoint).resolve()
    output_root = (repo / args.output_root).resolve()
    report_dir = (repo / args.report_dir).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for condition, test_data_name in CONDITIONS.items():
        condition_dir = output_root / condition
        workbook = condition_dir / "classification_results.xlsx"
        config_path = condition_dir / "evaluation_config.json"
        condition_dir.mkdir(parents=True, exist_ok=True)

        config = dict(base_config)
        config.update(
            {
                "experiment_name": f"geolife_vs_{condition}",
                "task": "dual_branch_classification",
                "data_name": "geolife_user_fixed_5s",
                "train_data_name": "geolife_user_fixed_5s",
                "val_data_name": "geolife_user_fixed_5s",
                "test_data_name": test_data_name,
                "output_dir": str(condition_dir.relative_to(repo)),
                "records_file": str(
                    (condition_dir / "records.xlsx").relative_to(repo)
                ),
                "test_only": "testset",
                "load_model": str(checkpoint.relative_to(repo)),
                "limit_size": None,
            }
        )
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if not (args.skip_existing and workbook.is_file()):
            print(f"\n=== Evaluating {condition}: {test_data_name} ===", flush=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--config",
                    str(config_path.relative_to(repo)),
                ],
                cwd=repo,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Evaluation failed for {condition} "
                    f"(exit {completed.returncode})"
                )

        result = parse_result(workbook)
        results[condition] = {
            "data_name": test_data_name,
            **result,
        }
        print(
            f"{condition}: accuracy={result['accuracy']:.6f}, "
            f"macro_f1={result['macro_f1']:.6f}",
            flush=True,
        )

    baseline = results["fixed_5s"]
    for result in results.values():
        result["accuracy_delta_vs_fixed_5s"] = (
            result["accuracy"] - baseline["accuracy"]
        )
        result["macro_f1_delta_vs_fixed_5s"] = (
            result["macro_f1"] - baseline["macro_f1"]
        )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    payload = {
        "protocol": "geolife-user-disjoint-variable-sampling-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit_before_results": commit,
        "base_config": str(base_config_path.relative_to(repo)),
        "checkpoint": str(checkpoint.relative_to(repo)),
        "checkpoint_sha256": sha256(checkpoint),
        "results": results,
    }
    json_path = report_dir / "geolife_variable_sampling_results.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = report_dir / "geolife_variable_sampling_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "condition",
            "data_name",
            "samples",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "accuracy_delta_vs_fixed_5s",
            "macro_f1_delta_vs_fixed_5s",
        ]
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for condition, result in results.items():
            writer.writerow(
                {
                    key: result.get(key)
                    for key in fieldnames
                    if key != "condition"
                }
                | {"condition": condition}
            )

    print(f"\nSaved: {json_path.relative_to(repo)}")
    print(f"Saved: {csv_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
