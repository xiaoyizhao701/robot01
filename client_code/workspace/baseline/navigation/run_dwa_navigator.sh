#!/usr/bin/env bash
# Run the standalone DWA navigator inside the client container.
# Publish a goal with:
#   ros2 topic pub /nav/goal geometry_msgs/PoseStamped \
#     "{header: {frame_id: base_link}, pose: {position: {x: 2.0, y: 2.5}}}"
set -e
source /opt/ros/humble/setup.bash
cd /workspace/baseline/navigation
export PYTHONPATH="/workspace/baseline/navigation:${PYTHONPATH:-}"
exec python3 dwa_navigator.py