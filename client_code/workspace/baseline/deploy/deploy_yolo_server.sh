#!/usr/bin/env bash
# Deploy the YOLO perception stack to the SERVER container (GPU inference) and
# start it.  Copies the perception code + trained model, patches the MJCF path
# for the server, and launches kele_detect with the 9-class YOLO backend.
#
# Usage: bash deploy/deploy_yolo_server.sh
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/examples/supermarket_sorting/perception"
DST="/workspace/supermarket_sorting_task/examples/supermarket_sorting/perception"

echo "[1/3] copy perception code + model to server"
docker cp "$SRC/." qy_run_server:$DST/
# the trained 9-class model (keep the checkout copy in the client bind-mount too)
docker cp "$REPO/yolo_model/best.pt" qy_run_server:$DST/checkpoints/products.pt

echo "[2/3] patch MJCF path for the server + kill old perception"
docker exec qy_run_server bash -lc "
  sed -i 's|MJCF_TASK_DIR = \"/workspace/baseline/examples/supermarket_sorting\"|MJCF_TASK_DIR = \"/workspace/supermarket_sorting_task/examples/supermarket_sorting\"|' $DST/kele_detect.py
  for p in \$(ps -eo pid,cmd | grep kele_detect | grep -v grep | awk '{print \$1}'); do kill -9 \$p; done 2>/dev/null
  true"

echo "[3/3] launch server-side GPU perception"
docker cp "$REPO/deploy/run_kele_server.sh" qy_run_server:/tmp/run_kele_server.sh
docker exec qy_run_server bash -lc 'setsid nohup bash /tmp/run_kele_server.sh > /tmp/kele_server.log 2>&1 < /dev/null & disown; sleep 14; grep -a "backend=" /tmp/kele_server.log | tail -1'

echo "done. detections on /kele/detections (ROS_DOMAIN_ID=96). log: /tmp/kele_server.log"