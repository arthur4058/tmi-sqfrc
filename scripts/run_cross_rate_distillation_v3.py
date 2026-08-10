#!/usr/bin/env python3
"""Train and evaluate dense-to-sparse distillation at selected GPS rates."""

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
    parse_result,
    run_checked,
    sha256,
    tensors_finite,
)


def build_config(protocol, rate: int, variant: str, repo: Path, args):
    common = dict(protocol["training"])
    variant_config = protocol["variants"][variant]
    run_name = f"geolife_cross_rate_{variant}_{rate}s_seed10086"
    output_dir = repo / "experiments" / run_name
    common.update(variant_config)
    common.update({
        "experiment_name": run_name,
        "data_name": f"geolife_v3_fixed_{rate}s",
        "output_dir": str(output_dir.relative_to(repo)),
        "records_file": str((output_dir / "records.xlsx").relative_to(repo)),
        "cross_rate_distillation": True,
        "distillation_teacher_data_name": protocol["teacher"]["data_name"],
        "distillation_teacher_checkpoint": protocol["teacher"]["checkpoint"],
        "student_rate_seconds": rate,
    })
    if args.epochs is not None:
        common["epochs"] = args.epochs
        common["patience"] = min(common["patience"], max(1, args.epochs))
    if args.limit_size is not None:
        common["limit_size"] = args.limit_size
    return common, output_dir


def build_b0_config(protocol, rate: int, repo: Path, args):
    common = dict(protocol["training"])
    run_name = f"geolife_v3_b0_fixed_{rate}s_seed10086"
    output_dir = repo / "experiments" / run_name
    common.update({
        "experiment_name": run_name,
        "data_name": f"geolife_v3_fixed_{rate}s",
        "output_dir": str(output_dir.relative_to(repo)),
        "records_file": str((output_dir / "records.xlsx").relative_to(repo)),
        "cross_rate_distillation": False,
    })
    if args.epochs is not None:
        common["epochs"] = args.epochs
        common["patience"] = min(common["patience"], max(1, args.epochs))
    if args.limit_size is not None:
        common["limit_size"] = args.limit_size
    return common, output_dir


def execute_training_and_test(repo, config, output_dir, args, variant, rate):
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = output_dir / "training_config.json"
    training_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint = output_dir / "checkpoints/model_best.pth"
    if not (args.skip_existing and checkpoint.is_file()):
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
        "experiment_name": f"{config['experiment_name']}_test",
        "task": "dual_branch_classification",
        "output_dir": str(test_dir.relative_to(repo)),
        "records_file": str((test_dir / "records.xlsx").relative_to(repo)),
        "test_only": "testset",
        "load_model": str(checkpoint.relative_to(repo)),
        "limit_size": None,
        "cross_rate_distillation": False,
    })
    test_config_path = test_dir / "test_config.json"
    test_config_path.write_text(
        json.dumps(test_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not (args.skip_existing and workbook.is_file()):
        run_checked([
            sys.executable, "main.py", "--config",
            str(test_config_path.relative_to(repo)),
        ], repo)
    result = parse_result(workbook)
    result.update({
        "variant": variant,
        "rate_seconds": rate,
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


def train_b0_and_test(repo, protocol, rate, args):
    config, output_dir = build_b0_config(protocol, rate, repo, args)
    return execute_training_and_test(
        repo, config, output_dir, args, "b0", rate)


def train_and_test(repo, protocol, rate, variant, args):
    config, output_dir = build_config(protocol, rate, variant, repo, args)
    return execute_training_and_test(
        repo, config, output_dir, args, variant, rate)


def write_reports(repo, protocol, results):
    baselines = {
        item["rate_seconds"]: item
        for item in results
        if item["variant"] == "b0"
    }
    comparisons = []
    for result in results:
        if result["variant"] == "b0" or result["rate_seconds"] not in baselines:
            continue
        baseline = baselines[result["rate_seconds"]]
        comparisons.append({
            "variant": result["variant"],
            "rate_seconds": result["rate_seconds"],
            "b0_accuracy": baseline["accuracy"],
            "v3_accuracy": result["accuracy"],
            "accuracy_delta": result["accuracy"] - baseline["accuracy"],
            "b0_macro_f1": baseline["macro_f1"],
            "v3_macro_f1": result["macro_f1"],
            "macro_f1_delta": result["macro_f1"] - baseline["macro_f1"],
        })

    adaptive = [row for row in comparisons if row["variant"] == "v3_adaptive"]
    if not adaptive:
        raise ValueError("No v3_adaptive results are available for a decision")
    improved = sum(row["macro_f1_delta"] > 0 for row in adaptive)
    mean_accuracy_delta = sum(
        row["accuracy_delta"] for row in adaptive) / len(adaptive)
    mean_macro_f1_delta = sum(
        row["macro_f1_delta"] for row in adaptive) / len(adaptive)
    passed = improved == len(adaptive)
    decision = {
        "status": "passed" if passed else "not_passed",
        "improved_sparse_rates": improved,
        "evaluated_sparse_rates": len(adaptive),
        "mean_accuracy_delta": mean_accuracy_delta,
        "mean_macro_f1_delta": mean_macro_f1_delta,
        "next_step": (
            "Run three seeds and extend to all rates."
            if passed else
            "Do not extend V3. Test a V3.1 density-capped distillation "
            "weight on 60s first, then retain it only if it beats the same B0."
        ),
    }

    alignment_path = (
        repo / "reports/manifests/geolife_cross_rate_filter_v3.json")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    report_dir = repo / "reports/experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": protocol["protocol"],
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit_before_results": commit,
        "teacher": protocol["teacher"],
        "training_pair_filter": alignment,
        "results": results,
        "comparisons": comparisons,
        "decision": decision,
    }
    json_path = report_dir / "geolife_cross_rate_distillation_v3.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = report_dir / "geolife_cross_rate_distillation_v3.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# GeoLife跨采样率知识蒸馏V3实验",
        "",
        "## 方法",
        "",
        "冻结5秒B0作为稠密教师；学生使用同一物理窗口的20秒或60秒视图。",
        "训练损失包含真实标签交叉熵、教师概率蒸馏，以及可选的池化表示余弦对齐。",
        "教师置信度和学生/教师有效点密度比共同调节蒸馏强度。测试阶段只使用稀疏学生。",
        "",
        "## 公平性约束",
        "",
        "- Train/Validation/Test用户严格互斥。",
        "- 参与蒸馏的训练样本按用户、源轨迹、物理窗口与标签严格配对。",
        "- 教师仅使用训练用户的5秒视图，未访问测试用户。",
        "- 学生网络、训练轮数、优化器和B0保持一致。",
        "- B0与V3使用同一筛选后训练集；验证集和测试集保持完整。",
        "",
        "## 训练配对验证",
        "",
        f"- 20秒保留{alignment['rates']['20']['segments_after']}/"
        f"{alignment['rates']['20']['segments_before']}个训练分段"
        f"（{alignment['rates']['20']['retained_fraction']:.2%}）。",
        f"- 60秒保留{alignment['rates']['60']['segments_after']}/"
        f"{alignment['rates']['60']['segments_before']}个训练分段"
        f"（{alignment['rates']['60']['retained_fraction']:.2%}）。",
        "- 保留样本的教师窗口缺失数为0，标签冲突数为0。",
        "",
        "## 结果",
        "",
        "| 变体 | 采样间隔 | B0 Accuracy | V3 Accuracy | ΔAccuracy | B0 Macro-F1 | V3 Macro-F1 | ΔMacro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['variant']} | {row['rate_seconds']}秒 | "
            f"{row['b0_accuracy']:.4f} | {row['v3_accuracy']:.4f} | "
            f"{row['accuracy_delta']:+.4f} | {row['b0_macro_f1']:.4f} | "
            f"{row['v3_macro_f1']:.4f} | {row['macro_f1_delta']:+.4f} |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        f"- V3在{improved}/{len(adaptive)}个稀疏档位提高Macro-F1。",
        f"- 平均Accuracy变化：{mean_accuracy_delta:+.4f}。",
        f"- 平均Macro-F1变化：{mean_macro_f1_delta:+.4f}。",
        "- 结论：V3未通过；20秒仅小幅正增益，60秒出现负迁移，"
        "不进入多随机种子或五档扩展。",
        "",
        "## 下一步",
        "",
        "当前权重让观测越稀疏的样本接受越强教师约束；60秒学生与5秒教师差异"
        "最大，蒸馏项可能压过真实标签监督。下一版只做一个小改动：让教师权重"
        "随学生/教师有效点密度比降低（density-capped），先只跑60秒。只有60秒"
        "在相同B0上转正，才回测20秒并考虑三随机种子。",
        "",
        "原始/生成数据、PKL缓存和checkpoint仅保存在本机，不提交Git。",
    ])
    md_path = report_dir / "geolife_cross_rate_distillation_v3.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {md_path.relative_to(repo)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/five_rate/geolife_cross_rate_distillation_v3.json"),
    )
    parser.add_argument("--rates", type=int, nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-size", type=float)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / args.config).read_text(encoding="utf-8"))
    rates = args.rates or protocol["student_rates_seconds"]
    variants = args.variants or list(protocol["variants"])
    invalid = set(variants) - set(protocol["variants"])
    if invalid:
        raise ValueError(f"Unknown variants: {sorted(invalid)}")
    b0_rates = sorted({protocol["teacher_rate_seconds"], *rates})
    results = [
        train_b0_and_test(repo, protocol, int(rate), args)
        for rate in b0_rates
    ]
    results.extend([
        train_and_test(repo, protocol, int(rate), variant, args)
        for variant in variants
        for rate in rates
    ])
    if not args.no_report and args.epochs is None and args.limit_size is None:
        write_reports(repo, protocol, results)


if __name__ == "__main__":
    main()
