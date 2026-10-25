# Supermarket Sorting Task

## 运行环境版本

- CUDA：12.8（Server 基础镜像为 `nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04`）
- Client：CUDA 12.8、PyTorch 2.7.1+cu128、ROS2 Humble、OpenCV、Ultralytics、MuJoCo
- ROS2：`ROS_DOMAIN_ID=99`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`

超市分拣比赛 ROS2 仿真环境。Server 运行 MuJoCo + 可选 3DGS 场景，发布相机、深度、
激光雷达、里程计和关节状态；选手将自己的 Baseline 挂载到固定 Client 容器中运行。

## 部署

宿主机需要 Docker、NVIDIA Driver（Linux >= 570.26）、NVIDIA Container Toolkit 和
NVIDIA GPU。cu128 wheel 的 GPU 推理支持 Turing 及更新架构（RTX 20/30/40/50 系）；
YOLO 在不支持的 GPU 上会自动回退为 CPU 推理，但这不使 3DGS Server 变为 CPU 兼容。

拉取 Server 和 Client 镜像：

```bash
docker pull crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server
docker pull crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client
```

允许运行窗口显示，并创建运行缓存：

```bash
xhost +local:docker
docker volume create supermarket_sorting_cache
```

## 固定 Baseline

Baseline 源码仓库：

```text
https://github.com/cyathea152-bit/supermarket_sorting_baseline
```

默认流程为：到达货架、视觉检测可乐、抓取、抬起、退出货架，随后结束并保持抓取，
不执行配送台流程。固定 Baseline 的目标可乐为 `product_032`，位于货架 D、L2、C2，
世界坐标约为 `[0.92, 3.243, 0.9235]`；D/L2 的其他位置没有可乐。

## 正式运行

正式 Server 使用随机商品和随机障碍物：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_FIXED_BASELINE=1 \
  -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 \
  -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

正式 Client 挂载选手 Baseline，启动后保持运行：

```bash
docker run -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /path/to/your_baseline:/workspace/baseline:rw \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client
```

选手程序通过 `docker exec` 在 Client 容器内启动。

### Server

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=1 \
  -e SUPERMARKET_SEED=11 \
  -e SUPERMARKET_TASKS=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

### Client

将 `/path/to/supermarket_sorting_baseline` 改成 Baseline 绝对路径：

```bash
docker run -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /path/to/supermarket_sorting_baseline:/workspace/baseline:rw \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client
```

### Baseline

```bash
docker exec -it supermarket_sorting_client \
  bash -lc 'cd /workspace/baseline && ./scripts/run_baseline.sh'
```

Baseline 检测目标来自 `/kele/detections` 的视觉世界坐标；检测不到时等待，不回退到
固定槽位坐标。替换权重时直接替换 Baseline 中的 `weights/kele.pt`，不需要修改 Client
镜像。

## 随机任务 Server

随机任务模式下，每次启动 45 个商品和通道内 5 个障碍物都会重新随机放置。障碍物按
底盘转弯扫掠半径和安全余量生成，拒绝重叠、贴墙、封死通路或无需绕行的布局。每个
纵向区段放置 1 块，形成连续蛇形避障；障碍物只随机改变平面偏航角。Server 使用
栅格 A* 验证货架入口至配送台入口存在可通行路线，并输出路径和绕行距离。

Baseline 的固定航点不是随机避障答案；正式任务中选手应使用
`/slamware_ros_sdk_server_node/scan` 自行实现路径规划和动态避障。

Server 也可以使用以下固定配置测试 Baseline：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e SUPERMARKET_FIXED_BASELINE=1 \
  -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 \
  -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

任务消息示例：

```json
{"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":2,"targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"},{"id":"item_run_a1b2c3d4e5f6_02","kind":"kele"}]}
```

`targets` 数组长度与 `count` 一致。每个 `id` 是本次 Server 进程专用的匿名 body 名，
不包含货架位置；`kind` 是商品类别。即使使用相同 `SUPERMARKET_SEED`，每次启动的
`run_prefix` 也会重新生成。可用以下命令查看当前任务：

```bash
ros2 topic echo --once /supermarket_sorting/task
```

## ROS2 话题

`ROS_DOMAIN_ID` 必须在 Server 和 Client 之间保持一致。

### Server 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/slamware_ros_sdk_server_node/odom` | `nav_msgs/msg/Odometry` | 底盘世界位姿和速度 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 动态 TF |
| `/slamware_ros_sdk_server_node/scan` | `sensor_msgs/msg/LaserScan` | 360 点二维激光雷达，默认 12 Hz |
| `/joint_states` | `sensor_msgs/msg/JointState` | 升降、头部、双臂和夹爪状态 |
| `/head_camera/color/image_raw` | `sensor_msgs/msg/Image` | 头部 RGB 图像，640x480 |
| `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部 RGB 内参 |
| `/head_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 头部深度图，mono16，毫米 |
| `/head_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部深度内参 |
| `/left_camera/color/image_raw` | `sensor_msgs/msg/Image` | 左腕 RGB 图像 |
| `/left_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 左腕相机内参 |
| `/right_camera/color/image_raw` | `sensor_msgs/msg/Image` | 右腕 RGB 图像 |
| `/right_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 右腕相机内参 |
| `/supermarket_sorting/task` | `std_msgs/msg/String` | JSON 任务清单，可靠、持久化发布 |

### Server 订阅

| Topic | Type | 控制格式 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `linear.x` 前进，`angular.z` 转向 |
| `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[slide_joint]` |
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[head_yaw_joint, head_pitch_joint]` |
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 6 轴和夹爪 |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 6 轴和夹爪 |

### Baseline 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/kele/detections` | `vision_msgs/msg/Detection3DArray` | 可乐世界坐标检测结果 |
| `/kele/result_image` | `sensor_msgs/msg/Image` | 检测可视化图 |

### ArUco 识别接口

Server 仓库保留 ArUco 调试节点。`<camera>` 为 `head`、`left` 或 `right`：

| Topic | Type | 说明 |
| --- | --- | --- |
| `/aruco/<camera>/ids` | `std_msgs/msg/Int32MultiArray` | 当前帧识别到的 ID |
| `/aruco/<camera>/poses` | `geometry_msgs/msg/PoseArray` | 标记相对相机位姿 |
| `/aruco/<camera>/detections` | `std_msgs/msg/String` | JSON 检测结果 |
| `/aruco/<camera>/result_image` | `sensor_msgs/msg/Image` | 调试图 |

`/joint_states` 顺序：

```text
slide_joint, head_yaw_joint, head_pitch_joint,
left_arm_joint1..left_arm_joint6, left_arm_eef_gripper_joint,
right_arm_joint1..right_arm_joint6, right_arm_eef_gripper_joint
```

夹爪命令中 `gripper=1.0` 表示张开，Baseline 抓可乐时使用 `gripper=0.08` 闭合。

## 参数说明

| 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `99` | Server 和 Client 通信域 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | ROS2 RMW 实现 |
| `MUJOCO_GL` | `glfw` | 图形窗口；无头环境可用 `egl` |
| `SUPERMARKET_HEADLESS` | `0` | `0` 显示窗口，`1` 无头 |
| `SUPERMARKET_ENABLE_RENDER` | `1` | 发布 RGB-D |
| `SUPERMARKET_ENABLE_LIDAR` | `1` | 发布二维雷达 |
| `SUPERMARKET_USE_GS` | `1` | 启用 3DGS |
| `SUPERMARKET_GS_SEQUENTIAL` | `1` | 逐相机渲染，降低显存峰值 |
| `SUPERMARKET_FIXED_BASELINE` | `0` | 固定 D/L2 中央单可乐布局 |
| `SUPERMARKET_RANDOMIZE` | `1` | 随机商品位置 |
| `SUPERMARKET_SEED` | `11` | 商品布局随机种子 |
| `SUPERMARKET_RANDOMIZE_OBSTACLES` | `1` | 随机障碍物位置 |
| `SUPERMARKET_OBSTACLE_SEED` | 可选 | 障碍物随机种子 |
| `SUPERMARKET_TASKS` | `all` | 任务类别或内部 ID 筛选 |
| `SUPERMARKET_BASELINE_WEIGHTS` | 可选 | Baseline 权重路径 |

## 主要文件

```text
Server: examples/supermarket_sorting/supermarket_sorting_server.py
Baseline: https://github.com/cyathea152-bit/supermarket_sorting_baseline
Baseline 控制: client_task_1.py
Baseline 感知: perception/kele_detect.py
Baseline 权重: weights/kele.pt
雷达 Topic: /slamware_ros_sdk_server_node/scan
```
