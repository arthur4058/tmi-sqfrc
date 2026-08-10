#!/usr/bin/env python3
"""Train/test PTMA-v2 at five matched rates and compare with frozen B0."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from run_five_rate_matched_experiments import (
    CLASS_NAMES,
    class_f1,
    parse_result,
    run_checked,
    sha256,
    tensors_finite,
)


def build_training_config(common: dict, rate: int, repo: Path):
    data_name = f"geolife_five_rate_fixed_{rate}s"
    run_name = f"geolife_matched_ptma_v2_fixed_{rate}s_seed10086"
    output_dir = repo / "experiments" / run_name
    config = dict(common)
    config.update({
        "experiment_name": run_name,
        "data_name": data_name,
        "output_dir": str(output_dir.relative_to(repo)),
        "records_file": str((output_dir / "records.xlsx").relative_to(repo)),
        "sampling_quality_reliability": False,
        "physical_time_multiscale": True,
    })
    return config, output_dir


def train_and_test(repo: Path, common: dict, rate: int,
                   skip_existing: bool) -> dict:
    config, output_dir = build_training_config(common, rate, repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = output_dir / "matched_training_config.json"
    training_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint = output_dir / "checkpoints" / "model_best.pth"
    if not (skip_existing and checkpoint.is_file()):
        run_checked([
            sys.executable, "main.py", "--config",
            str(training_config.relative_to(repo)),
        ], repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not tensors_finite(payload):
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
        run_checked([
            sys.executable, "main.py", "--config",
            str(test_config_path.relative_to(repo)),
        ], repo)
    if not workbook.is_file():
        raise FileNotFoundError(workbook)

    result = parse_result(workbook)
    result.update({
        "variant": "ptma_v2",
        "rate_seconds": rate,
        "data_name": config["data_name"],
        "checkpoint": str(checkpoint.relative_to(repo)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_finite": True,
        "training_config": str(training_config.relative_to(repo)),
        "test_workbook": str(workbook.relative_to(repo)),
    })
    print(
        f"RESULT PTMA-v2 {rate}s: accuracy={result['accuracy']:.6f}, "
        f"macro_f1={result['macro_f1']:.6f}", flush=True)
    return result


def load_b0_results(repo: Path, protocol: dict) -> dict[int, dict]:
    path = repo / protocol["baseline_results_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = {
        int(item["rate_seconds"]): item
        for item in payload["results"]
        if item["variant"] == "b0"
    }
    missing = set(protocol["rates_seconds"]) - set(results)
    if missing:
        raise ValueError(f"B0 results missing rates: {sorted(missing)}")
    return results


def write_reports(repo: Path, protocol: dict, results: list[dict]) -> Path:
    rates = [int(rate) for rate in protocol["rates_seconds"]]
    b0 = load_b0_results(repo, protocol)
    ptma = {int(item["rate_seconds"]): item for item in results}
    comparisons = []
    for rate in rates:
        comparisons.append({
            "rate_seconds": rate,
            "b0_accuracy": b0[rate]["accuracy"],
            "ptma_v2_accuracy": ptma[rate]["accuracy"],
            "accuracy_delta": ptma[rate]["accuracy"] - b0[rate]["accuracy"],
            "b0_macro_f1": b0[rate]["macro_f1"],
            "ptma_v2_macro_f1": ptma[rate]["macro_f1"],
            "macro_f1_delta": ptma[rate]["macro_f1"] - b0[rate]["macro_f1"],
        })

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True).stdout.strip()
    report_dir = repo / "reports" / "experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": protocol["protocol"],
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit_before_results": commit,
        "seed": protocol["training"]["seed"],
        "physical_time_windows_seconds": protocol["training"][
            "physical_time_windows_seconds"],
        "baseline_results_path": protocol["baseline_results_path"],
        "results": results,
        "comparisons": comparisons,
    }
    json_path = report_dir / "geolife_physical_time_multiscale_v2.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    csv_path = report_dir / "geolife_physical_time_multiscale_v2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)

    mean_b0 = sum(item["b0_macro_f1"] for item in comparisons) / len(rates)
    mean_ptma = sum(item["ptma_v2_macro_f1"] for item in comparisons) / len(rates)
    improved = sum(item["macro_f1_delta"] > 0 for item in comparisons)
    sparse_improved = all(
        next(item for item in comparisons if item["rate_seconds"] == rate)[
            "macro_f1_delta"] > 0
        for rate in (30, 60)
    )
    success = (
        mean_ptma - mean_b0 >= 0.01
        and improved >= 4
        and sparse_improved
    )
    lines = [
        "# GeoLife物理时间多尺度适配模块（版本2）实验结果",
        "",
        "## 研究问题",
        "",
        "原基线按相邻GPS点提取局部运动模式。同样数量的点在不同采样间隔下对应不同真实时长，导致表示含义不一致。",
        "版本2使用原始时间戳，在30、60、120秒三个真实时间邻域提取历史运动上下文，并通过尺度注意力融合；零初始化残差保留原基线特征。",
        "",
        "## 公平对比设置",
        "",
        "- 数据、用户互斥划分、五档采样视图均与B0完全相同",
        "- 每档分别训练并在同档测试：5、10、20、30、60秒",
        f"- 随机种子：{protocol['training']['seed']}",
        f"- 最大轮次：{protocol['training']['epochs']}；Early Stopping patience：{protocol['training']['patience']}",
        "- 唯一实验变量：是否启用PTMA-v2模块",
        "",
        "## 主结果",
        "",
        "| 采样间隔 | B0 Accuracy | V2 Accuracy | ΔAccuracy | B0 Macro-F1 | V2 Macro-F1 | ΔMacro-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['rate_seconds']}秒 | {item['b0_accuracy']:.4f} | "
            f"{item['ptma_v2_accuracy']:.4f} | {item['accuracy_delta']:+.4f} | "
            f"{item['b0_macro_f1']:.4f} | {item['ptma_v2_macro_f1']:.4f} | "
            f"{item['macro_f1_delta']:+.4f} |")

    lines.extend([
        "",
        "## 各类别F1",
        "",
        "| 模型 | 采样间隔 | Walk | Bike | Bus | Car | Train |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, source in (("B0", b0), ("PTMA-v2", ptma)):
        for rate in rates:
            values = [class_f1(source[rate], name) for name in CLASS_NAMES]
            rendered = " | ".join(
                "N/A" if value is None else f"{value:.4f}"
                for value in values)
            lines.append(f"| {variant} | {rate}秒 | {rendered} |")

    lines.extend([
        "",
        "## 汇总判断",
        "",
        f"- B0五档平均Macro-F1：{mean_b0:.4f}",
        f"- PTMA-v2五档平均Macro-F1：{mean_ptma:.4f}",
        f"- 平均变化：{mean_ptma - mean_b0:+.4f}",
        f"- 提升档位：{improved}/5",
        f"- 30秒与60秒均提升：{'是' if sparse_improved else '否'}",
        f"- 预设成功标准（平均至少+0.01、至少4/5档提升、30/60秒均提升）：{'通过' if success else '未通过'}",
        "",
        "## 后续建议",
        "",
        (
            "当前模块达到预设标准。下一步先做三个随机种子确认稳定性，再进行消融实验。"
            if success else
            "当前模块未达到预设标准，不应包装成有效创新。应根据分档和分类别结果调整时间尺度或融合方式。"
        ),
        "",
        "本机生成数据、PKL、训练日志和checkpoint不提交Git。",
    ])
    markdown = report_dir / "geolife_physical_time_multiscale_v2.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {markdown.relative_to(repo)}")
    return markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/five_rate/geolife_ptma_v2_seed10086.json"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rates", type=int, nargs="+")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / args.config).read_text(encoding="utf-8"))
    rates = args.rates or protocol["rates_seconds"]
    results = [
        train_and_test(repo, protocol["training"], int(rate),
                       args.skip_existing)
        for rate in rates
    ]
    if set(map(int, rates)) == set(map(int, protocol["rates_seconds"])):
        write_reports(repo, protocol, results)


if __name__ == "__main__":
    main()
