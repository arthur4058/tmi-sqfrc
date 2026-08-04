#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/yc/miniconda3/envs/tmi-sqfrc/bin/python}"
s4_script="$repo_root/tmi/data_preprocess/s4_trajectory_feature_calculation_with_CPD.py"
views_root="$repo_root/data/geolife_published_sampling_views"
logs_root="$repo_root/data/logs"

mkdir -p "$logs_root"

run_s4() {
    local input_trjs="$1"
    local input_labels="$2"
    local output_dir="$3"
    local log_name="$4"
    mkdir -p "$output_dir"
    "$python_bin" "$s4_script" \
        --trjs_path "$input_trjs" \
        --labels_path "$input_labels" \
        --n_class 5 \
        --save_dir "$output_dir" \
        --kde_bw 1 \
        --kde_kernel epa \
        --mask_mode ep \
        --trj_mask_mode kde \
        --mask_ratio 0.3 \
        --min_n_points 4 \
        --disable_percentile_filter \
        --seed 42 \
        > "$logs_root/$log_name.log" 2>&1
    grep "Running time" "$logs_root/$log_name.log"
}

for condition in fixed_5s fixed_30s fixed_60s variable_5_60s
do
    feature_name="geolife_published_${condition}"
    run_s4 \
        "$views_root/$condition/test_trjs.npy" \
        "$views_root/$condition/test_labels.npy" \
        "$repo_root/data/${feature_name}_features/test" \
        "s4_published_${condition}_test"
done
