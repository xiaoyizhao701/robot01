# Supermarket Sorting Task Example

This directory contains the supermarket scene, ROS2 server, baseline client, and local assets.

Use the [repository root README](../../README.md) for Docker commands, server/client startup, ROS2 topic tables, and baseline control usage.

Main files:

```text
supermarket_sorting_server.py   # starts the MuJoCo/ROS2 simulation server
supermarket_sorting_client.py   # baseline pick-and-place client
perception/aruco_detect.py       # starts head/left/right ArUco detection nodes
retail_competition_layout.json  # shelf slot layout and product metadata
mjcf/retail_competition.xml     # MuJoCo scene
models/                         # meshes, textures, and 3DGS assets
```

The server publishes the simulated 2-D lidar by default. Start all three ArUco
camera nodes with:

```bash
source /opt/ros/humble/setup.bash
python3 examples/supermarket_sorting/perception/aruco_detect.py
```
