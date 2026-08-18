"""Train and evaluate V2 at additional random seeds on the 60-second view."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/v2_multiseed_configs"


B0_TEMPLATE = {
    "task": "dual_branch_classification_from_scratch",
    "data_class": "trajectory_with_feature",
    "use_separate_val": True,
    "epochs": 200,
    "val_interval": 1,
    "val_ratio": 0.0,
    "batch_size": 64,
    "lr": 0.001,
    "optimizer": "RAdam",
    "patience": 40,
    "input_type": "50%noise",
    "motion_features": [3, 4, 5, 8],
    "sampling_quality_feature": 0,
    "sampling_quality_hidden_dim": 16,
    "gpu": "0",
    "num_workers": 0,
    "console": True,
    "print_interval": 100,
    "trajectory_branch_hyperparams": (
        "feat_dim=2;max_len=200;d_model=64;n_heads=8;num_layers=4;"
        "dim_feedforward=256;dropout=0.1;pos_encoding=fixed;"
        "activation=gelu;norm=BatchNorm;freeze=false"
    ),
    "feature_branch_hyperparams": (
        "feat_dim=4;max_len=200;d_model=128;n_heads=16;num_layers=1;"
        "dim_feedforward=512;dropout=0.1;pos_encoding=learnable;"
        "activation=gelu;norm=LayerNorm;freeze=false"
    ),
    "data_name": "geolife_five_rate_fixed_60s",
    "sampling_quality_reliability": False,
    "key_metric": "accuracy",
    "protocol": "geolife-low-rate-three-expert-v2-multiseed",
}


def read_template(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build_seed_configs(seed):
    configs = {}
    definitions = (
        (
            "b0", dict(B0_TEMPLATE),
            f"geolife_matched_b0_fixed_60s_seed{seed}",
        ),
        (
            "four_feature",
            read_template("configs/consensus/geolife_feature_expert_60s.json"),
            f"geolife_60s_b2_feature_seed{seed}",
        ),
        (
            "motion",
            read_template("configs/low_rate/geolife_motion_expert_60s_seed10086.json"),
            f"geolife_motion_expert_60s_seed{seed}",
        ),
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, config, experiment_name in definitions:
        output_dir = ROOT / "experiments" / experiment_name
        config.update({
            "seed": int(seed),
            "experiment_name": experiment_name,
            "output_dir": str(output_dir),
            "records_file": str(output_dir / "records.xlsx"),
        })
        path = CONFIG_DIR / f"{experiment_name}.json"
        path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        configs[name] = path
    return configs


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}); inspect {log_path}"
        )


def train_if_needed(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    checkpoint = output_dir / "checkpoints/model_best.pth"
    if checkpoint.exists():
        print(f"[skip] checkpoint exists: {checkpoint}", flush=True)
        return
    print(f"[train] {config['experiment_name']}", flush=True)
    run_logged(
        [sys.executable, "main.py", "--config", str(config_path)],
        output_dir / "multiseed_training.log",
    )
    if not checkpoint.exists():
        raise RuntimeError(f"training produced no best checkpoint: {checkpoint}")


def run_seed(seed):
    configs = build_seed_configs(seed)
    for name in ("b0", "four_feature", "motion"):
        train_if_needed(configs[name])
    output_dir = ROOT / f"experiments/geolife_three_expert_60s_seed{seed}"
    print(f"[evaluate] V2 seed {seed}", flush=True)
    run_logged(
        [
            sys.executable, "-m", "scripts.evaluate_three_expert_consensus",
            "--rate", "60", "--seed", str(seed),
        ],
        output_dir / "multiseed_evaluation.log",
    )
    result_path = output_dir / "three_expert_consensus_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "seed": seed,
        "weights": result["weights"],
        "baseline": result["test"]["baseline"],
        "consensus": result["test"]["consensus"],
        "delta": result["test_delta"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for seed in args.seeds:
        run_seed(seed)


if __name__ == "__main__":
    main()
