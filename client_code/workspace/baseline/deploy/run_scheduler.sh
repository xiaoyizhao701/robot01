#!/bin/bash
cd /workspace/baseline
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=96
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/navigation:/workspace/baseline:$(python3 -c "import sys;print(':'.join(p for p in sys.path if '/opt/ros' in p))")
exec python3 -u /workspace/baseline/task_scheduler.py --max-targets "${1:-5}" --client dwa
