# Supermarket Sorting Task

## 运行环境版本

- CUDA：12.8（Docker 基础镜像为 `nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04`）
- PyTorch：2.7.1+cu128
- torchvision：0.22.1+cu128

超市分拣比赛 ROS2 仿真环境。server 运行 MuJoCo + 可选 3DGS 场景，发布相机、深度、激光雷达、里程计和关节状态；参赛 client 通过 ROS2 话题控制 MMK2 机器人完成货架取货和配送台放置。

## 部署

宿主机需要 Docker、NVIDIA Driver（Linux >= 570.26）、NVIDIA Container Toolkit 和 NVIDIA GPU。cu128
wheel 的 GPU 推理支持 Turing 及更新架构（RTX 20/30/40/50 系）；Pascal/GTX 10、Volta
等较旧架构不在当前镜像的 GPU 加速支持范围。YOLO 会自动回退为 CPU 推理，速度会降低，
但这不使 3DGS/EGL Server 变为 CPU 兼容。拉取比赛镜像：

```bash
docker pull crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:main
```

如需从源码重新构建镜像：

```bash
docker build -t crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:main .
```

允许运行窗口显示，并创建运行缓存：

```bash
xhost +local:docker
docker volume create supermarket_sorting_cache
```

## 固定 Baseline

这一组命令用于验证 baseline。货架 D 的 L2 层只有中间 C2 是可乐，baseline 会通过视觉检测抓取该可乐并完成一次搬运。

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
  -e SUPERMARKET_FIXED_BASELINE=1 \
  -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 \
  -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:main \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

### Baseline Client

baseline client 同时启动可乐视觉检测节点和头部、左右腕部 3 个 ArUco 识别节点。抓取目标来自 `/kele/detections` 的视觉世界坐标；检测不到时等待，不回退到固定槽位坐标。

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_baseline \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:main \
  bash -lc 'cd /workspace/supermarket_sorting_task && ./docker/run_simplified_baseline.sh'
```

如只需启动全部 ArUco 识别节点，可在与 Server 相同的 `ROS_DOMAIN_ID` 下运行：

```bash
source /opt/ros/humble/setup.bash
python3 examples/supermarket_sorting/perception/aruco_detect.py
```

默认同时订阅头部、左腕和右腕 RGB 相机。可用 `--cameras head` 只开启指定相机，或用
`--no-tf`、`--no-result-image` 关闭 TF 和调试图发布。场景标记使用
`DICT_4X4_50`、ID `0~44`，边长为 `0.03 m`。

## 随机任务 Server

随机任务模式下，每次启动 45 个商品和通道内 5 个障碍物都会重新随机放置。Server 只负责下发任务、运行仿真并提供传感器和机器人控制接口，不包含裁判和计分功能。

障碍物不是无约束地随机撒点。Server 按 MMK2 底盘完整转弯扫掠半径加 `0.05 m`
安全余量生成布局，拒绝障碍物重叠、贴墙、封死通路或无需明显绕行的结果。每个正式
布局在下、下中、中、中上、上 5 个纵向区段各放置 1 块；它以经过验证的蛇形模板为骨架，
再在每个位置做小范围随机扰动。相邻区段必须横向错开至少 `0.40 m`、纵向间隔至少
`0.40 m`，形成连续蛇形避障，而不是把多块障碍聚在一起后整体绕开。障碍物会随机改变
平面偏航角（绕 `Z` 轴），不使用会让箱体倾倒的 `Y` 轴旋转。Server 通过 `0.05 m` 栅格
A* 验证货架侧入口至配送台入口存在机器人可通行路线，且相对空通道至少增加 `0.30 m`
绕行距离；启动日志会输出生成尝试次数、路径长度和绕行距离。

README 的 Baseline 命令启用 `SUPERMARKET_RANDOMIZE_OBSTACLES=1`。Baseline 的固定航点
不是随机避障答案；正式赛题中选手应使用
`/slamware_ros_sdk_server_node/scan` 自行实现路径规划和动态避障。

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
  -e SUPERMARKET_TASKS=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:main \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

任务消息示例：

```json
{"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":2,"targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"},{"id":"item_run_a1b2c3d4e5f6_02","kind":"kele"}]}
```

实际 `targets` 数组长度与 `count` 一致。每个 `id` 是本次 Server 进程专用的匿名
MuJoCo body 名，不包含货架、层数或列号；`kind` 是商品业务类别。即使使用相同
`SUPERMARKET_SEED` 复现物理布局，`run_prefix` 也会在每次 Server 启动时重新生成。
`SUPERMARKET_TASKS` 可按商品类别或中性内部 ID（如 `product_001`）筛选，正式使用建议按类别筛选。
Client 可以在相同 `ROS_DOMAIN_ID` 下查看当前任务：

```bash
ros2 topic echo --once /supermarket_sorting/task
```

## ROS2 话题

`ROS_DOMAIN_ID` 必须在 server 和 client 之间保持一致。下表只列比赛相关接口，不包含 `/rosout`、`/parameter_events` 等 ROS2 自动话题。

### Server 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/slamware_ros_sdk_server_node/odom` | `nav_msgs/msg/Odometry` | 底盘世界位姿和速度 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 动态 TF，主要为 `odom -> base_link` |
| `/slamware_ros_sdk_server_node/scan` | `sensor_msgs/msg/LaserScan` | 360 点二维激光雷达，默认 12 Hz，`frame_id=laser` |
| `/joint_states` | `sensor_msgs/msg/JointState` | 升降、头部、双臂和夹爪关节状态 |
| `/head_camera/color/image_raw` | `sensor_msgs/msg/Image` | 头部 RGB 图像，默认 640x480 |
| `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部 RGB 相机内参 |
| `/head_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 头部深度图，`mono16`，单位毫米 |
| `/head_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部深度相机内参 |
| `/left_camera/color/image_raw` | `sensor_msgs/msg/Image` | 左腕 RGB 图像 |
| `/left_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 左腕相机内参 |
| `/right_camera/color/image_raw` | `sensor_msgs/msg/Image` | 右腕 RGB 图像 |
| `/right_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 右腕相机内参 |
| `/supermarket_sorting/task` | `std_msgs/msg/String` | JSON 任务清单，包含 `schema_version`、本局 `run_prefix`、`count` 和匿名目标 `{id, kind}`；可靠、持久化发布 |

### Server 订阅

| Topic | Type | 控制格式 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `linear.x` 前进速度，`angular.z` 角速度 |
| `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[slide_joint]` |
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[head_yaw_joint, head_pitch_joint]` |
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[joint1, joint2, joint3, joint4, joint5, joint6, gripper]` |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[joint1, joint2, joint3, joint4, joint5, joint6, gripper]` |

### Baseline 视觉节点发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/kele/detections` | `vision_msgs/msg/Detection3DArray` | 可乐检测结果，位姿为世界坐标系 `frame_id=world` |
| `/kele/result_image` | `sensor_msgs/msg/Image` | 检测可视化图，仅调试用 |

### ArUco 识别节点发布

下表中的 `<camera>` 为 `head`、`left` 或 `right`。`ids` 与 `poses` 的数组下标一一对应，位姿位于对应相机坐标系；同时还会发布 `aruco_<camera>_<id>` TF。

| Topic | Type | 说明 |
| --- | --- | --- |
| `/aruco/<camera>/ids` | `std_msgs/msg/Int32MultiArray` | 当前帧识别到的 ArUco ID，升序排列 |
| `/aruco/<camera>/poses` | `geometry_msgs/msg/PoseArray` | 各标记相对相机的 6D 位姿 |
| `/aruco/<camera>/detections` | `std_msgs/msg/String` | 含 ID、平移和四元数的 JSON 检测结果 |
| `/aruco/<camera>/result_image` | `sensor_msgs/msg/Image` | 带边框、ID 和坐标轴的调试图 |

`/joint_states` 关节顺序如下：

```text
slide_joint
head_yaw_joint
head_pitch_joint
left_arm_joint1
left_arm_joint2
left_arm_joint3
left_arm_joint4
left_arm_joint5
left_arm_joint6
left_arm_eef_gripper_joint
right_arm_joint1
right_arm_joint2
right_arm_joint3
right_arm_joint4
right_arm_joint5
right_arm_joint6
right_arm_eef_gripper_joint
```

夹爪命令中 `gripper=1.0` 表示张开，baseline 抓可乐时使用 `gripper=0.08` 闭合。

## 参数说明

| 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `99` | ROS2 通信域，server/client 必须一致 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | ROS2 RMW 实现 |
| `MUJOCO_GL` | `glfw` | 图形窗口运行；无头环境可改为 `egl` |
| `SUPERMARKET_HEADLESS` | `0` | `0` 显示窗口，`1` 无头运行 |
| `SUPERMARKET_ENABLE_RENDER` | `1` | 发布相机 RGB-D 数据 |
| `SUPERMARKET_ENABLE_LIDAR` | `1` | 启用二维激光雷达并发布 `/slamware_ros_sdk_server_node/scan` |
| `SUPERMARKET_USE_GS` | `1` | 启用 3D Gaussian Splatting 渲染 |
| `SUPERMARKET_GS_SEQUENTIAL` | `1` | 逐相机执行 3DGS 渲染，降低多相机峰值显存 |
| `SUPERMARKET_RANDOMIZE` | `1` | 是否随机打乱 45 个商品位置 |
| `SUPERMARKET_SEED` | 可选 | 指定随机种子，使随机布局可复现；baseline 验证使用 `11` |
| `SUPERMARKET_RANDOMIZE_OBSTACLES` | `1` | 是否随机放置通道内 5 个障碍物 |
| `SUPERMARKET_OBSTACLE_SEED` | 可选 | 独立指定障碍物随机种子；未指定时由 `SUPERMARKET_SEED` 派生 |
| `SUPERMARKET_TASKS` | `all` | 下发任务目标；可用逗号分隔的商品类别或中性内部 ID（如 `product_001`）筛选 |
| `--device`（`kele_detect.py`） | `auto` | YOLO 使用兼容 GPU；不支持时自动回退 CPU，可指定 `cpu` |

主要文件：

```text
examples/supermarket_sorting/supermarket_sorting_server.py   # 仿真 server
examples/supermarket_sorting/supermarket_sorting_client.py   # baseline 控制 client
examples/supermarket_sorting/perception/kele_detect.py       # baseline 可乐视觉检测
examples/supermarket_sorting/perception/aruco_detect.py      # 头部和双腕 ArUco 识别
examples/supermarket_sorting/retail_competition_layout.json  # 货架布局和商品元数据
```
