#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/yc/miniconda3/envs/tmi-sqfrc/bin/python}"

cd "$repo_root"
"$python_bin" -m tmi.data_preprocess.variable_sampling \
    --data_dir data/geolife_user_split \
    --output_dir data/geolife_five_rate_views \
    --manifest_path reports/manifests/geolife_five_rate_seed42.json \
    --conditions fixed_5s,fixed_10s,fixed_20s,fixed_30s,fixed_60s \
    --seed 42 \
    --window_seconds 300 \
    --stride_seconds 150 \
    --min_points 5

"$python_bin" scripts/validate_five_rate_dataset.py \
    --views-root data/geolife_five_rate_views \
    --output reports/manifests/geolife_five_rate_validation.json \
    --seed 42
