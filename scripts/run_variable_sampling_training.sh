#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/yc/miniconda3/envs/tmi-sqfrc/bin/python}"
cd "$repo_root"
exec "$python_bin" main.py \
    --config configs/variable_sampling/geolife_user_fixed5_seed10086.json
