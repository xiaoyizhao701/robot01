#!/bin/bash
cd /workspace/tools
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=96
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/workspace/tools:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
export DISPLAY=:2
exec python3 ros_dashboard.py
