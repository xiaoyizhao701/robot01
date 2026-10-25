#!/usr/bin/env bash
# Bring up the supermarket sim server + client stack on ROS_DOMAIN_ID=96,
# deploying the locally-patched server-side files (which live in the image and
# are NOT bind-mounted).
#
# Usage: bash deploy/deploy_stack.sh
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN=96

echo "[1/3] server container (qy_run_server)"
docker rm -f qy_run_server 2>/dev/null || true
docker run -d --gpus all --network host --ipc host \
  --name qy_run_server \
  -e DISPLAY=:2 -e ROS_DOMAIN_ID=$DOMAIN -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw -e SUPERMARKET_HEADLESS=0 -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 -e SUPERMARKET_USE_GS=1 \
  -e SUPERMARKET_GS_SCENE_ONLY=0 -e SUPERMARKET_SENSOR_HZ=6 \
  -e SUPERMARKET_WRIST_RENDER_DIVISOR=3 \
  -e SUPERMARKET_RGB_CAMS=0,1,2 -e SUPERMARKET_DEPTH_CAMS=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=1 -e SUPERMARKET_OBSTACLE_SEED=62 \
  -e SUPERMARKET_FIXED_BASELINE=1 -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_TASKS=product_001,product_002,product_003,product_004,product_005 -e SUPERMARKET_TARGET_RTF=1.0 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e TORCH_CUDA_ARCH_LIST=8.6 \
  -e XAUTHORITY=/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v supermarket_sorting_cache:/root/.cache \
  -v /home/qy/.Xauthority:/root/.Xauthority:ro \
  supermarket_sorting:server_v3 \
  bash -lc 'sleep 6; cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 /workspace/supermarket_tools/fast_server.py'

echo "[2/3] deploy patched server-side files"
# wait for the container to exist, then copy before fast_server imports them
for i in $(seq 1 30); do docker exec qy_run_server true 2>/dev/null && break; sleep 1; done
docker cp "$REPO/obstacle_layout.py" \
  qy_run_server:/workspace/supermarket_sorting_task/examples/supermarket_sorting/obstacle_layout.py
docker cp "$REPO/examples/supermarket_sorting/models/mjcf/mobile_chassis/mmk2/mmk2.xml" \
  qy_run_server:/workspace/supermarket_sorting_task/examples/supermarket_sorting/models/mjcf/mobile_chassis/mmk2/mmk2.xml
docker cp "$REPO/examples/supermarket_sorting/models/mjcf/manipulator/airbot_play/airbot_play_dependencies.xml" \
  qy_run_server:/workspace/supermarket_sorting_task/examples/supermarket_sorting/models/mjcf/manipulator/airbot_play/airbot_play_dependencies.xml
docker cp "$REPO/examples/supermarket_sorting/models/mjcf/mobile_chassis/mmk2/mmk2_dependencies.xml" \
  qy_run_server:/workspace/supermarket_sorting_task/examples/supermarket_sorting/models/mjcf/mobile_chassis/mmk2/mmk2_dependencies.xml
docker cp "$REPO/deploy/mmk2_ros2.py" \
  qy_run_server:/workspace/supermarket_sorting_task/examples/ros2/mmk2_ros2.py

echo "[3/3] client container (qy_run_client)"
if ! docker inspect qy_run_client >/dev/null 2>&1; then
  docker run -d --gpus all --network host --ipc host \
    --name qy_run_client \
    -e ROS_DOMAIN_ID=$DOMAIN -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e DISPLAY=:2 -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$REPO":/workspace/baseline \
    -v "/home/qy/supermarket_tools":/workspace/tools_qy \
    supermarket_sorting:client_v3 bash -lc 'sleep infinity'
else
  docker start qy_run_client >/dev/null
fi

# kill any leftover pick-place clients / schedulers so they don't drive the robot
docker exec qy_run_client bash -lc 'pkill -9 -f task_scheduler 2>/dev/null; pkill -9 -f supermarket_sorting_client_dwa 2>/dev/null; true' 2>/dev/null || true

echo "done. server needs ~30s to init. spawn check:"
echo "  docker exec qy_run_client bash -lc 'source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=$DOMAIN RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; ros2 topic echo --once /slamware_ros_sdk_server_node/odom'"

# Set sim time-flow to MAX (target_rtf=0) so the GPU runs as fast as it can.
sleep 30
docker exec qy_run_client bash -lc "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=$DOMAIN RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; \
  ros2 topic pub -1 /supermarket_sorting/control std_msgs/msg/String \"data: '{\\\"action\\\":\\\"set_rate\\\",\\\"target_rtf\\\":0}'\" >/dev/null 2>&1 && echo 'rate=MAX set'"
