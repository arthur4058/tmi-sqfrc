#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/yc/miniconda3/envs/tmi-sqfrc/bin/python}"

cd "$repo_root"
"$python_bin" -m tmi.data_preprocess.variable_sampling \
    --data_dir data/geolife_user_split \
    --output_dir data/geolife_published_sampling_views \
    --manifest_path reports/manifests/geolife_published_sampling_seed42.json \
    --conditions fixed_5s,fixed_30s,fixed_60s,variable_5_60s \
    --seed 42 \
    --window_seconds 300 \
    --stride_seconds 150 \
    --min_points 5

"$python_bin" scripts/validate_published_sampling_dataset.py \
    --views-root data/geolife_published_sampling_views \
    --legacy-root data/geolife_sampling_views \
    --output reports/manifests/geolife_published_sampling_validation.json \
    --seed 42
