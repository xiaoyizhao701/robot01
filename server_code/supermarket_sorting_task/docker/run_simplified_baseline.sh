#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
cd "$repo_root"

python3 examples/supermarket_sorting/perception/kele_detect.py \
  --backend "${SUPERMARKET_DETECTOR_BACKEND:-yolo}" \
  --device "${SUPERMARKET_DETECTOR_DEVICE:-auto}" &
detector_pid=$!

cleanup() {
  kill "$detector_pid" 2>/dev/null || true
  wait "$detector_pid" 2>/dev/null || true
}
trap cleanup EXIT

python3 examples/supermarket_sorting/supermarket_sorting_client_simplified.py
