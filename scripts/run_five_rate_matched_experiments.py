#!/usr/bin/env python3
"""Train and test B0/M1 separately at five matched GPS sampling rates."""

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
import torch


CLASS_NAMES = ("Walk", "Bike", "Bus", "Car", "Train")


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


def average_row(report: pd.DataFrame) -> pd.Series:
    if "class" in report.columns:
        labels = report["class"].astype(str).str.lower()
        rows = report[labels.str.contains("avg") | labels.str.contains("total")]
        if not rows.empty:
            return rows.iloc[-1]
    for index in reversed(report.index.tolist()):
        if "avg" in str(index).lower() or "total" in str(index).lower():
            return report.loc[index]
    return report.iloc[-1]


def parse_result(workbook: Path) -> dict:
    report = pd.read_excel(
        workbook, sheet_name="Classification Report", index_col=0
    )
    average = average_row(report)
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
        name = str(row.get("class", index))
        if "avg" in name.lower() or "total" in name.lower():
            continue
        per_class[name] = {
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
    }


def tensors_finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_finite(item) for item in value)
    return True


def run_checked(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}"
        )


def build_training_config(common: dict, rate: int, variant: str, repo: Path):
    data_name = f"geolife_five_rate_fixed_{rate}s"
    run_name = f"geolife_matched_{variant}_fixed_{rate}s_seed10086"
    output_dir = repo / "experiments" / run_name
    config = dict(common)
    config.update({
        "experiment_name": run_name,
        "data_name": data_name,
        "output_dir": str(output_dir.relative_to(repo)),
        "records_file": str((output_dir / "records.xlsx").relative_to(repo)),
        "sampling_quality_reliability": variant == "m1",
    })
    return config, output_dir


def train_and_test(
        repo: Path, common: dict, rate: int, variant: str,
        skip_existing: bool) -> dict:
    config, output_dir = build_training_config(common, rate, variant, repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = output_dir / "matched_training_config.json"
    training_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint = output_dir / "checkpoints" / "model_best.pth"

    if not (skip_existing and checkpoint.is_file()):
        run_checked(
            [
                sys.executable,
                "main.py",
                "--config",
                str(training_config.relative_to(repo)),
            ],
            repo,
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if not tensors_finite(checkpoint_payload):
        raise RuntimeError(f"Non-finite checkpoint: {checkpoint}")

    test_dir = output_dir / "matched_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    workbook = test_dir / "classification_results.xlsx"
    test_config = dict(config)
    test_config.update({
        "experiment_name": f"{config['experiment_name']}_matched_test",
        "task": "dual_branch_classification",
        "output_dir": str(test_dir.relative_to(repo)),
        "records_file": str((test_dir / "records.xlsx").relative_to(repo)),
        "test_only": "testset",
        "load_model": str(checkpoint.relative_to(repo)),
        "limit_size": None,
    })
    test_config_path = test_dir / "matched_test_config.json"
    test_config_path.write_text(
        json.dumps(test_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not (skip_existing and workbook.is_file()):
        run_checked(
            [
                sys.executable,
                "main.py",
                "--config",
                str(test_config_path.relative_to(repo)),
            ],
            repo,
        )
    if not workbook.is_file():
        raise FileNotFoundError(workbook)

    result = parse_result(workbook)
    result.update({
        "variant": variant,
        "rate_seconds": rate,
        "data_name": config["data_name"],
        "checkpoint": str(checkpoint.relative_to(repo)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_finite": True,
        "training_config": str(training_config.relative_to(repo)),
        "test_workbook": str(workbook.relative_to(repo)),
    })
    print(
        f"RESULT {variant} {rate}s: accuracy={result['accuracy']:.6f}, "
        f"macro_f1={result['macro_f1']:.6f}",
        flush=True,
    )
    return result


def class_f1(result: dict, name: str):
    for key, values in result["per_class"].items():
        if key.lower() == name.lower():
            return values["f1"]
    return None


def write_reports(repo: Path, protocol: dict, results: list[dict]) -> Path:
    report_dir = repo / "reports" / "experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    by_key = {
        (item["variant"], item["rate_seconds"]): item
        for item in results
    }
    rates = protocol["rates_seconds"]

    comparisons = []
    for rate in rates:
        b0 = by_key[("b0", rate)]
        m1 = by_key[("m1", rate)]
        comparisons.append({
            "rate_seconds": rate,
            "b0_accuracy": b0["accuracy"],
            "m1_accuracy": m1["accuracy"],
            "accuracy_delta": m1["accuracy"] - b0["accuracy"],
            "b0_macro_f1": b0["macro_f1"],
            "m1_macro_f1": m1["macro_f1"],
            "macro_f1_delta": m1["macro_f1"] - b0["macro_f1"],
        })

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validation_path = (
        repo / "reports/manifests/geolife_five_rate_validation.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    payload = {
        "protocol": protocol["protocol"],
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit_before_results": commit,
        "seed": protocol["training"]["seed"],
        "dataset_validation": validation,
        "results": results,
        "comparisons": comparisons,
    }

    json_path = report_dir / "geolife_five_rate_matched_seed10086.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = report_dir / "geolife_five_rate_matched_seed10086.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(comparisons[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# GeoLife五档采样率匹配训练与测试结果",
        "",
        "## 实验结论",
        "",
        "本实验对5、10、20、30、60秒五种固定GPS采样间隔分别训练并测试。",
        "B0与M1在每一档使用完全相同的数据、用户划分和训练超参数；本报告不包含跨采样率或未见采样率测试。",
        "",
        "## 实验协议",
        "",
        f"- 协议：`{protocol['protocol']}`",
        "- 数据集：GeoLife 1.3，五类交通方式",
        "- 用户划分：Train/Validation/Test严格互斥",
        "- 物理窗口：300秒；步长：150秒",
        "- 采样算子：等物理时间分箱，每个非空分箱保留第一个真实GPS点，不插值",
        "- 采样间隔：5、10、20、30、60秒",
        f"- 模型随机种子：{protocol['training']['seed']}",
        f"- 最大训练轮次：{protocol['training']['epochs']}；Early Stopping patience：{protocol['training']['patience']}",
        "- 主指标：Macro-F1；辅助指标：Accuracy和各类别F1",
        "",
        "## 数据验证",
        "",
        f"- 验证状态：`{validation['status']}`",
        "- 用户互斥：通过",
        "- 五档标签、用户ID、源轨迹ID和窗口ID对齐：通过",
        "- 五档均由同一5秒真实点视图确定性生成：通过",
        "- 平均点密度随5→10→20→30→60秒严格下降：通过",
        "- 坐标插值或合成：无",
        "",
        "## B0与M1主结果",
        "",
        "| 采样间隔 | B0 Accuracy | M1 Accuracy | ΔAccuracy | B0 Macro-F1 | M1 Macro-F1 | ΔMacro-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['rate_seconds']}秒 | {item['b0_accuracy']:.4f} | "
            f"{item['m1_accuracy']:.4f} | {item['accuracy_delta']:+.4f} | "
            f"{item['b0_macro_f1']:.4f} | {item['m1_macro_f1']:.4f} | "
            f"{item['macro_f1_delta']:+.4f} |"
        )

    lines.extend([
        "",
        "## 各类别F1",
        "",
        "| 模型 | 采样间隔 | Walk | Bike | Bus | Car | Train |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in ("b0", "m1"):
        for rate in rates:
            result = by_key[(variant, rate)]
            values = [class_f1(result, name) for name in CLASS_NAMES]
            rendered = " | ".join(
                "N/A" if value is None else f"{value:.4f}"
                for value in values
            )
            lines.append(f"| {variant.upper()} | {rate}秒 | {rendered} |")

    mean_b0 = sum(item["b0_macro_f1"] for item in comparisons) / len(comparisons)
    mean_m1 = sum(item["m1_macro_f1"] for item in comparisons) / len(comparisons)
    improved = sum(item["macro_f1_delta"] > 0 for item in comparisons)
    lines.extend([
        "",
        "## 汇总结论",
        "",
        f"- B0五档平均Macro-F1：{mean_b0:.4f}",
        f"- M1五档平均Macro-F1：{mean_m1:.4f}",
        f"- M1平均变化：{mean_m1 - mean_b0:+.4f}",
        f"- M1在{improved}/5个采样率上提高Macro-F1",
        "",
        "判断标准：只有当M1在多数采样率、五档平均Macro-F1及稀疏档位上均表现更好，才能认为当前模块形成稳定改进。",
        "如果仅少数档位提升，本报告将如实记录，并据此调整模块，而不重复包装为新的基线版本。",
        "",
        "## 复现入口",
        "",
        "```bash",
        "bash scripts/generate_five_rate_matched_dataset.sh",
        "bash scripts/generate_five_rate_features.sh",
        "python scripts/run_five_rate_matched_experiments.py --skip-existing",
        "```",
        "",
        "原始数据、生成NPY/PKL、checkpoint和训练日志仅保存在本机，不提交Git。",
    ])
    markdown_path = report_dir / "geolife_five_rate_matched_seed10086.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {markdown_path.relative_to(repo)}")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/five_rate/geolife_matched_seed10086.json"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / args.config).read_text(encoding="utf-8"))
    results = []
    for variant in protocol["variants"]:
        for rate in protocol["rates_seconds"]:
            results.append(train_and_test(
                repo,
                protocol["training"],
                int(rate),
                variant,
                args.skip_existing,
            ))
    write_reports(repo, protocol, results)


if __name__ == "__main__":
    main()
