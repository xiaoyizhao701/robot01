# Supermarket Sorting Task 仓库功能说明

## 1. 仓库概述

本仓库是一套面向超市商品分拣比赛的 ROS2 仿真环境，基于 DISCOVERSE、MuJoCo 和 3D Gaussian Splatting（3DGS）构建。

系统用于控制 MMK2 移动双臂机器人从货架抓取商品，绕过场地障碍，并将商品搬运到配送台。Server 负责下发任务、运行仿真以及提供传感器和控制接口，不包含裁判和计分功能。

整体数据流如下：

```text
MuJoCo + 3DGS 仿真 Server
  ├─ 发布 RGB-D、里程计、关节状态和 TF
  ├─ 接收底盘、升降柱、头部、机械臂和夹爪指令
  └─ 发布结构化任务清单
                         ↑ ROS2
视觉检测节点 ────────────→ Baseline 控制 Client
```

## 2. 主要功能

### 2.1 超市竞赛场景仿真

场景文件位于：

```text
examples/supermarket_sorting/mjcf/retail_competition.xml
```

场景包含：

- 5 个货架，每个货架有 3 层、每层 3 个货位；
- 45 件可交互商品；
- 9 类商品：可乐、脉动、和味道、薯片、口香糖、三明治、纸巾、苹果和橙子，每类 5 件；
- 起始区、取货区、配送台、围墙和 5 个障碍箱；
- 每个货位对应一个独立的 ArUco 标记；
- 商品具有碰撞模型和自由关节，可以被抓取、搬运、掉落或碰倒。

### 2.2 MuJoCo 与 3DGS 混合渲染

仿真 Server 使用 MuJoCo 处理机器人运动、接触和商品动力学，并使用 3DGS 渲染机器人、商品和超市背景。

支持以下运行方式：

- GLFW 图形窗口模式；
- EGL 无头运行模式；
- 开启或关闭相机渲染；
- 开启或关闭 3DGS；
- 发布默认分辨率为 640×480 的头部和腕部相机图像。

主要入口：

```text
examples/supermarket_sorting/supermarket_sorting_server.py
```

### 2.3 ROS2 机器人接口

Server 将 MMK2 仿真封装为 ROS2 节点。

主要发布话题：

| Topic | 功能 |
| --- | --- |
| `/joint_states` | 升降柱、头部、双臂和夹爪关节状态 |
| `/slamware_ros_sdk_server_node/odom` | 底盘世界位姿和速度 |
| `/tf` | 动态坐标变换 |
| `/head_camera/color/image_raw` | 头部 RGB 图像 |
| `/head_camera/aligned_depth_to_color/image_raw` | 头部深度图，单位为毫米 |
| `/left_camera/color/image_raw` | 左腕 RGB 图像 |
| `/right_camera/color/image_raw` | 右腕 RGB 图像 |

主要控制话题：

| Topic | 功能 |
| --- | --- |
| `/cmd_vel` | 底盘线速度和角速度控制 |
| `/spine_forward_position_controller/commands` | 升降柱位置控制 |
| `/head_forward_position_controller/commands` | 头部偏航和俯仰控制 |
| `/left_arm_forward_position_controller/commands` | 左臂六轴关节和夹爪控制 |
| `/right_arm_forward_position_controller/commands` | 右臂六轴关节和夹爪控制 |

ROS2 桥接实现位于：

```text
examples/ros2/mmk2_ros2.py
```

### 2.4 商品位置随机化

Server 可以在每局开始时将 45 件商品重新打乱到不同货位，并根据商品自身高度和目标货架层修正 Z 坐标。

支持：

- 完全随机布局；
- 使用固定随机种子复现实验；
- 调试时固定指定商品；
- 按商品类别或中性内部 ID（`product_001` 等）筛选任务目标；任务消息不会暴露这些内部 ID。

### 2.5 任务下发

Server 启动后通过 `/supermarket_sorting/task` 发布 JSON 格式的任务清单，包含协议版本 `schema_version`、本局随机前缀 `run_prefix`、目标数量 `count`，以及由匿名 body ID 和商品类别构成的 `targets`。匿名 ID 不包含货架位置，并在每次 Server 启动时重新生成。

该话题使用可靠、持久化 QoS。任务只需发布一次，后启动的 Client 仍然可以取得当前任务。默认下发全部商品，也可以使用 `SUPERMARKET_TASKS` 按商品类别或中性内部 ID 筛选任务目标。

Server 不进行流程判断、计分、扣分或结果保存，也不向参赛 Client 暴露比赛重置服务。

### 2.6 可乐 RGB-D 视觉定位

可乐检测节点订阅头部 RGB、深度、相机内参、关节状态和里程计，完成以下处理：

1. 在 RGB 图像中检测可乐；
2. 使用检测框中心及深度图恢复相机坐标；
3. 根据机器人底盘、升降柱和头部状态计算相机世界位姿；
4. 将检测点转换为世界坐标；
5. 发布三维检测结果和可视化图像。

输出话题：

| Topic | 功能 |
| --- | --- |
| `/kele/detections` | 世界坐标系下的可乐三维检测结果 |
| `/kele/result_image` | 带检测框和坐标的调试图像 |

检测节点提供三种后端：

- `yolo`：使用仓库附带的 `kele.pt` 权重；
- `blob`：基于亮度阈值和连通域检测，适合简单背景；
- `gt`：将布局真值投影到图像，用于验证坐标转换链。

主要文件：

```text
examples/supermarket_sorting/perception/kele_detect.py
examples/supermarket_sorting/perception/backends.py
```

### 2.7 自动抓取搬运 Baseline

Baseline Client 实现了一次完整的可乐抓取和配送状态机：

```text
沿固定路线导航到货架
→ 等待可乐视觉坐标
→ 使用逆运动学计算右臂姿态
→ 底盘缓慢向前进给
→ 闭合夹爪
→ 升降柱抬起商品
→ 退出货架
→ 沿固定路线导航到配送台
→ 降低商品并松开夹爪
```

控制程序还包含：

- MMK2 正运动学和逆运动学；
- 底盘速度及加速度限制；
- 关节目标平滑插值；
- 关节反馈和动作到位判断；
- 多帧视觉结果中值融合；
- 视觉目标可达性筛选；
- 启动时请求重置当前比赛。

主要入口：

```text
examples/supermarket_sorting/supermarket_sorting_client.py
```

### 2.8 感知数据集和 YOLO 训练工具

`perception` 目录提供完整的可乐检测训练辅助工具：

- `gen_dataset.py`：利用 3DGS 仿真自动生成 YOLO 数据集；
- `train_yolo.py`：基于生成的数据集微调 YOLOv8s；
- `validate_cam_bridge.py`：验证像素、深度、相机坐标和世界坐标的转换精度。

数据生成支持：

- 随机相机位姿；
- 背景域随机化；
- 投影标签或掩码标签；
- 自动划分训练集和验证集；
- 输出标准 YOLOv8 目录及 `data.yaml`。

### 2.9 Docker 部署

仓库提供基于 Ubuntu 22.04、CUDA 11.8 和 ROS2 Humble 的 Docker 环境，集成：

- MuJoCo；
- PyTorch 和 torchvision；
- ROS2 Humble；
- CycloneDDS；
- OpenCV 与 cv_bridge；
- Ultralytics YOLO；
- 3D Gaussian Splatting 渲染依赖。

根目录 `README.md` 给出了预构建镜像、源码构建、Server 和 Baseline Client 的启动命令。

## 3. 当前实现边界

该仓库提供的是竞赛仿真环境和示范 Baseline，不是完整的通用超市分拣方案。目前存在以下边界：

- Baseline 仅识别和抓取可乐；
- 一次运行只完成一次搬运，进入 `DONE` 状态后不会继续抓取下一件商品；
- 导航依赖硬编码路点，没有集成 SLAM、全局路径规划或动态避障；
- 场景中名为 `dynamic_obstacle` 的障碍物目前实际是固定几何体；
- 超市场景明确关闭了激光雷达仿真；
- 仓库虽然包含 45 个 ArUco 标记资源，但没有现成的 ArUco 检测节点；
- 另外 8 类商品没有附带检测模型和抓取策略；
- Baseline 主要面向固定可乐验证场景，不能直接解决正式随机布局下的连续多商品分拣任务。

## 4. 总结

本仓库已经形成以下完整基础闭环：

```text
竞赛场景搭建
→ 机器人和传感器仿真
→ ROS2 控制接口
→ RGB-D 商品定位
→ 机械臂抓取与底盘搬运
→ 配送台放置
→ Server 下发任务
→ 感知数据生成与模型训练
```

在此基础上，参赛者仍需要开发多类别商品识别、自主导航与避障、通用抓取规划、失败恢复和连续多商品任务调度，才能形成完整的正式比赛方案。
