"""Measure B0, relation-expert and SORF-TMI inference complexity.

The benchmark uses actual 30-second and 60-second test tensors. FLOPs are
forward-pass operator FLOPs reported by PyTorch's FlopCounterMode at batch 1.
Latency and peak allocated memory use batch 64 with inputs resident on CUDA.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as training_main
from scripts.run_v74_five_rate_multiseed import (
    ROOT,
    RateGeneralizedRelationNet,
    anchor_count,
    load_split,
    normalize,
    select_aligned,
)
from tmi.datasets import dataset


class SORFInference(nn.Module):
    """Exact probability-fusion inference path used by the final method."""

    def __init__(self, b0, expert, b0_temperature, expert_temperature, weight):
        super().__init__()
        self.b0 = b0
        self.expert = expert
        self.b0_temperature = float(b0_temperature)
        self.expert_temperature = float(expert_temperature)
        self.weight = float(weight)

    def forward(self, x1, mask1, x2, mask2, relation_x, relation_mask):
        b0_logits = self.b0(x1, mask1, x2, mask2)
        expert_logits, _ = self.expert(relation_x, relation_mask)
        b0_probability = torch.softmax(b0_logits / self.b0_temperature, dim=1)
        expert_probability = torch.softmax(
            expert_logits / self.expert_temperature, dim=1)
        return (
            (1.0 - self.weight) * b0_probability
            + self.weight * expert_probability
        )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)


def tensor_bytes(model: nn.Module) -> int:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return int(parameter_bytes + buffer_bytes)


def move_inputs(inputs, device):
    return tuple(value.to(device, non_blocking=False) for value in inputs)


def invoke(model, inputs):
    output = model(*inputs)
    if isinstance(output, tuple):
        return output[0]
    return output


def flop_count(model: nn.Module, inputs) -> int:
    one = tuple(value[:1] for value in inputs)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        invoke(model, one)
    return int(counter.get_total_flops())


def benchmark(model: nn.Module, cpu_inputs, device: str, warmup: int,
              iterations: int, repeats: int) -> dict:
    if not device.startswith("cuda"):
        raise ValueError("this benchmark requires CUDA for comparable timing")
    model = model.to(device).eval()
    inputs = move_inputs(cpu_inputs, device)
    batch_size = int(inputs[0].shape[0])
    torch.cuda.empty_cache()

    with torch.inference_mode():
        for _ in range(warmup):
            invoke(model, inputs)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        invoke(model, inputs)
        torch.cuda.synchronize()
        peak_bytes = int(torch.cuda.max_memory_allocated())

        batch_latency_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                invoke(model, inputs)
            end.record()
            end.synchronize()
            batch_latency_ms.append(float(start.elapsed_time(end) / iterations))

        flops = flop_count(model, inputs)

    trainable, total = count_parameters(model)
    result = {
        "parameters": total,
        "trainable_parameters": trainable,
        "model_tensor_mib": tensor_bytes(model) / (1024 ** 2),
        "forward_flops_per_sample": flops,
        "forward_gflops_per_sample": flops / 1e9,
        "benchmark_batch_size": batch_size,
        "batch_latency_ms_median": statistics.median(batch_latency_ms),
        "batch_latency_ms_mean": statistics.mean(batch_latency_ms),
        "batch_latency_ms_std": statistics.stdev(batch_latency_ms)
        if len(batch_latency_ms) > 1 else 0.0,
        "per_sample_latency_ms_median": statistics.median(batch_latency_ms)
        / batch_size,
        "throughput_samples_per_second": 1000.0 * batch_size
        / statistics.median(batch_latency_ms),
        "peak_allocated_mib": peak_bytes / (1024 ** 2),
        "latency_repeats_ms": batch_latency_ms,
    }
    model.to("cpu")
    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def prepare_b0(rate: int, seed: int, batch_size: int, device: str):
    base = ROOT / f"experiments/geolife_matched_b0_fixed_{rate}s_seed{seed}"
    checkpoint = base / "checkpoints/model_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = json.loads((base / "configuration.json").read_text(encoding="utf-8"))
    output = ROOT / f"experiments/complexity_b0_{rate}s_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config.update({
        "task": "dual_branch_classification",
        "test_only": "testset",
        "load_model": str(checkpoint),
        "output_dir": str(output),
        "records_file": str(output / "records.xlsx"),
        "gpu": "0" if device.startswith("cuda") else "-1",
        "batch_size": batch_size,
        "num_workers": 0,
    })
    dataset.config = config
    pipeline = training_main.TrainingPipeline(config)
    pipeline.setup_data()
    pipeline.prepare_data_loaders()
    pipeline.setup_dl_model()
    model = pipeline.model.to("cpu").eval()
    batch = next(iter(pipeline.test_loader))
    # Collate order is trajectory, feature, trajectory mask, feature mask;
    # model order is trajectory, trajectory mask, feature, feature mask.
    inputs = tuple(
        batch[index][:batch_size].contiguous().cpu()
        for index in (0, 2, 1, 3)
    )
    pipeline.model = None
    del pipeline, batch
    gc.collect()
    torch.cuda.empty_cache()
    return model, inputs


def prepare_expert(rate: int, seed: int, batch_size: int):
    path = ROOT / f"experiments/geolife_v74_generalized_{rate}s_seed{seed}/model_best.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = RateGeneralizedRelationNet(
        anchor_count(rate), width=settings["width"], layers=settings["layers"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test = load_split(rate, "test")
    values, mask, _, _ = select_aligned(
        test, seed, checkpoint["mean"], checkpoint["std"])
    inputs = (
        torch.from_numpy(values[:batch_size]).contiguous(),
        torch.from_numpy(mask[:batch_size]).contiguous(),
    )
    return model.eval(), inputs


def fusion_settings(rate: int, seed: int):
    report = json.loads(
        (ROOT / "reports/experiments/geolife_v74_five_rate_multiseed.json")
        .read_text(encoding="utf-8")
    )
    matches = [row for row in report["results"]
               if row["rate_seconds"] == rate and row["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"expected one result for rate={rate}, seed={seed}")
    row = matches[0]
    return {
        "b0_temperature": row["temperatures"]["b0"],
        "expert_temperature": row["temperatures"]["relation_expert"],
        "fusion_weight": row["validation"]["fusion"]["weight"],
    }


def benchmark_rate(rate: int, args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    b0, b0_inputs = prepare_b0(
        rate, args.seed, args.batch_size, args.device)
    expert, expert_inputs = prepare_expert(rate, args.seed, args.batch_size)
    settings = fusion_settings(rate, args.seed)

    b0_result = benchmark(
        b0, b0_inputs, args.device, args.warmup, args.iterations, args.repeats)
    expert_result = benchmark(
        expert, expert_inputs, args.device,
        args.warmup, args.iterations, args.repeats)
    combined = SORFInference(
        b0, expert, settings["b0_temperature"],
        settings["expert_temperature"], settings["fusion_weight"])
    sorf_inputs = (*b0_inputs, *expert_inputs)
    sorf_result = benchmark(
        combined, sorf_inputs, args.device,
        args.warmup, args.iterations, args.repeats)

    if sorf_result["parameters"] != (
        b0_result["parameters"] + expert_result["parameters"]
    ):
        raise AssertionError("SORF parameter count is not the sum of both experts")
    if sorf_result["forward_flops_per_sample"] < max(
        b0_result["forward_flops_per_sample"],
        expert_result["forward_flops_per_sample"],
    ):
        raise AssertionError("SORF FLOPs unexpectedly below an individual expert")
    return {
        "rate_seconds": rate,
        "anchors": anchor_count(rate),
        "seed": args.seed,
        "fusion": settings,
        "input_shapes": {
            "b0": [list(value.shape) for value in b0_inputs],
            "relation_expert": [list(value.shape) for value in expert_inputs],
        },
        "models": {
            "b0": b0_result,
            "relation_expert": expert_result,
            "sorf_tmi": sorf_result,
        },
    }


def markdown_report(payload: dict) -> str:
    lines = [
        "# SORF-TMI model complexity benchmark",
        "",
        f"> Device: {payload['device']}",
        f"> PyTorch: {payload['pytorch_version']}",
        f"> CUDA runtime: {payload['cuda_runtime']}",
        f"> Batch size: {payload['benchmark']['batch_size']}",
        f"> Warm-up: {payload['benchmark']['warmup']} iterations; "
        f"measurement: {payload['benchmark']['repeats']} repeats × "
        f"{payload['benchmark']['iterations']} iterations",
        "",
        "FLOPs use PyTorch `FlopCounterMode` for one forward sample. Latency and "
        "peak allocated CUDA memory use actual test tensors already resident on "
        "the GPU; data loading and host-to-device transfer are excluded.",
        "",
        "| Rate | Model | Parameters | GFLOPs/sample | Batch latency (ms) | "
        "Latency/sample (ms) | Throughput (sample/s) | Peak memory (MiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "b0": "B0",
        "relation_expert": "Relation expert",
        "sorf_tmi": "SORF-TMI",
    }
    for rate in payload["rates"]:
        for key in ("b0", "relation_expert", "sorf_tmi"):
            row = rate["models"][key]
            lines.append(
                f"| {rate['rate_seconds']} s | {labels[key]} | "
                f"{row['parameters']:,} | "
                f"{row['forward_gflops_per_sample']:.4f} | "
                f"{row['batch_latency_ms_median']:.3f} | "
                f"{row['per_sample_latency_ms_median']:.4f} | "
                f"{row['throughput_samples_per_second']:.1f} | "
                f"{row['peak_allocated_mib']:.1f} |"
            )
    lines.extend((
        "",
        "## Interpretation",
        "",
        "SORF-TMI executes B0 and the relation expert and then fuses their "
        "probabilities, so its parameter count and compute are necessarily above "
        "B0. The relation expert is cheaper at 60 seconds because six real "
        "anchors generate fewer pairwise tokens than the eleven anchors at 30 "
        "seconds. The benchmark supports an accuracy-efficiency trade-off claim; "
        "it does not support a claim that SORF-TMI is faster than B0.",
        "",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=int, default=[30, 60])
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="reports/experiments/sorf_tmi_model_complexity_seed10086.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper benchmark")
    os.chdir(ROOT)
    device_name = torch.cuda.get_device_name(torch.device(args.device))
    payload = {
        "protocol": "sorf-tmi-complexity-v1",
        "device": device_name,
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "benchmark": {
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "inputs_resident_on_gpu": True,
            "includes_data_loading": False,
            "flop_counter": "torch.utils.flop_counter.FlopCounterMode",
        },
        "rates": [benchmark_rate(rate, args) for rate in args.rates],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(
        markdown_report(payload), encoding="utf-8")
    print(output)
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
