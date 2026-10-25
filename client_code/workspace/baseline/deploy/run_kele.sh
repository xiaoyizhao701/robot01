#!/bin/bash
cd /workspace/baseline/examples/supermarket_sorting
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=96
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/workspace/baseline/examples/supermarket_sorting:/workspace/baseline:$(python3 -c "import sys;print(':'.join(p for p in sys.path if '/opt/ros' in p))")
export DISPLAY=:2
exec python3 -u perception/kele_detect.py --backend yolo --device cpu --conf 0.45