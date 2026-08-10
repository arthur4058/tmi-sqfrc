#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/yc/miniconda3/envs/tmi-sqfrc/bin/python}"
s4_script="$repo_root/tmi/data_preprocess/s4_trajectory_feature_calculation_with_CPD.py"
views_root="$repo_root/data/geolife_five_rate_views"
logs_root="$repo_root/data/logs/cross_rate_distillation_v3"

cd "$repo_root"
mkdir -p "$logs_root"

for rate in 5 20 60; do
    condition="fixed_${rate}s"
    for split in train val test; do
        output_dir="$repo_root/data/geolife_v3_fixed_${rate}s_features/$split"
        mkdir -p "$output_dir"
        "$python_bin" "$s4_script" \
            --trjs_path "$views_root/$condition/${split}_trjs.npy" \
            --labels_path "$views_root/$condition/${split}_labels.npy" \
            --pair_ids_path "$views_root/$condition/${split}_pair_ids.npy" \
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
            > "$logs_root/s4_${condition}_${split}.log" 2>&1
        grep "Running time" "$logs_root/s4_${condition}_${split}.log"
    done
done

"$python_bin" scripts/filter_cross_rate_train_pairs.py
"$python_bin" scripts/validate_cross_rate_pairs.py
