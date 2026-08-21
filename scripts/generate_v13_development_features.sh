#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/yc/miniconda3/envs/tmi-sqfrc/bin/python}"
split_root="$repo_root/data/geolife_v13_dev_split_60s"
aug_root="$repo_root/data/geolife_v13_dev_60s_augmented"
feature_root="$repo_root/data/geolife_v13_dev_fixed_60s_features"
log_root="$repo_root/data/logs/v13_dev"
s4="$repo_root/tmi/data_preprocess/s4_trajectory_feature_calculation_with_CPD.py"

cd "$repo_root"
mkdir -p "$log_root"
"$python_bin" scripts/prepare_v13_development_data.py
"$python_bin" -m tmi.data_preprocess.s3_data_augmentation \
    --data_dir "$split_root" \
    --save_dir "$aug_root" \
    --random_seed 42 > "$log_root/s3_train_a.log" 2>&1

run_s4() {
    local trjs="$1"
    local labels="$2"
    local output="$3"
    local log="$4"
    mkdir -p "$output"
    "$python_bin" "$s4" \
        --trjs_path "$trjs" \
        --labels_path "$labels" \
        --n_class 5 \
        --save_dir "$output" \
        --kde_bw 1 \
        --kde_kernel epa \
        --mask_mode ep \
        --trj_mask_mode kde \
        --mask_ratio 0.3 \
        --min_n_points 4 \
        --disable_percentile_filter \
        --seed 42 > "$log_root/$log.log" 2>&1
    grep "Running time" "$log_root/$log.log"
}

run_s4 \
    "$aug_root/train_trjs_augmented.npy" \
    "$aug_root/train_labels_augmented.npy" \
    "$feature_root/train" s4_train_a
run_s4 \
    "$split_root/val_trjs.npy" \
    "$split_root/val_labels.npy" \
    "$feature_root/val" s4_dev_a
run_s4 \
    "$split_root/test_trjs.npy" \
    "$split_root/test_labels.npy" \
    "$feature_root/test" s4_dev_a_test_alias

"$python_bin" - <<'PY'
from pathlib import Path
import numpy as np
root = Path("data/geolife_v13_dev_fixed_60s_features")
for split in ("train", "val", "test"):
    labels = np.load(root / split / "noise_multi_feature_seg_labels.npy")
    print(split, len(labels), np.unique(labels, return_counts=True))
PY
