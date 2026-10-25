#!/bin/bash
# Server-side YOLO perception launcher (GPU). Runs in qy_run_server.
cd /workspace/supermarket_sorting_task/examples/supermarket_sorting
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=96
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/workspace/supermarket_sorting_task/examples/supermarket_sorting:/workspace/supermarket_sorting_task:$(python3 -c "import sys;print(chr(58).join(p for p in sys.path if 'ros' in p))")
export DISPLAY=:2
exec python3 -u perception/kele_detect.py --backend yolo --device cuda --conf 0.45