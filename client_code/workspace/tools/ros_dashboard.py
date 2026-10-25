#!/usr/bin/env python3
"""Compact visual dashboard for the supermarket ROS 2 simulation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState, LaserScan
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray


WINDOW = "Supermarket ROS Dashboard"
CANVAS_W, CANVAS_H = 960, 980
TILE_W, TILE_H = 300, 200
DISPLAY_FPS = float(os.getenv("SUPERMARKET_DASHBOARD_FPS", "12"))
IMAGE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
RESET_BUTTON = (775, 647, 935, 690)
RATE_BUTTONS = {
    0.5: (350, 647, 415, 690),
    1.0: (423, 647, 488, 690),
    2.0: (496, 647, 561, 690),
    0.0: (569, 647, 644, 690),
}
SEED_MINUS_BUTTON = (350, 702, 390, 742)
SEED_PLUS_BUTTON = (397, 702, 437, 742)
RANDOM_BUTTON = (447, 702, 547, 742)
GS_BUTTON = (555, 702, 643, 742)
APPLY_BUTTON = (651, 702, 755, 742)
MOVE_BUTTONS = {
    "forward": (92, 682, 160, 716),
    "left": (20, 720, 88, 754),
    "stop": (92, 720, 160, 754),
    "right": (164, 720, 232, 754),
    "back": (92, 758, 160, 792),
    "auto": (164, 758, 232, 792),
}

# --- Navigation overlay: arena map + goal controls ------------------------
# World bounds (from retail_competition.xml) — floor x in [-2.5, 2.5], y in [-3.75, 3.75].
WORLD_X_MIN, WORLD_X_MAX = -2.5, 2.5
WORLD_Y_MIN, WORLD_Y_MAX = -3.75, 3.75

MAP_PANEL = (640, 12, 948, 620)  # x0, y0, x1, y1 full panel
_MAP_MARGIN = 10
_MAP_IN_X0, _MAP_IN_Y0 = MAP_PANEL[0] + _MAP_MARGIN, MAP_PANEL[1] + _MAP_MARGIN
_MAP_IN_X1, _MAP_IN_Y1 = MAP_PANEL[2] - _MAP_MARGIN, MAP_PANEL[3] - _MAP_MARGIN
_MAP_SCALE = min((_MAP_IN_X1 - _MAP_IN_X0) / (WORLD_X_MAX - WORLD_X_MIN),
                 (_MAP_IN_Y1 - _MAP_IN_Y0) / (WORLD_Y_MAX - WORLD_Y_MIN))
MAP_X0 = (_MAP_IN_X0 + _MAP_IN_X1) / 2 - _MAP_SCALE * (WORLD_X_MAX - WORLD_X_MIN) / 2
MAP_X1 = MAP_X0 + _MAP_SCALE * (WORLD_X_MAX - WORLD_X_MIN)
MAP_Y0 = (_MAP_IN_Y0 + _MAP_IN_Y1) / 2 - _MAP_SCALE * (WORLD_Y_MAX - WORLD_Y_MIN) / 2
MAP_Y1 = MAP_Y0 + _MAP_SCALE * (WORLD_Y_MAX - WORLD_Y_MIN)

GOAL_SHELF_BUTTON = (636, 12, 840, 42)   # click -> goal at the shelf picking zone
GOAL_TABLE_BUTTON = (844, 12, 944, 42)   # click -> goal at the delivery table
GOAL_MAX_RANGE = 2.8                     # laser points kept for the obstacle overlay (m)

# Quick preset goals (safe approach points, not inside obstacles).
GOAL_SHELF = (0.852, 2.475)   # shelf approach lane, front of the shelf row
GOAL_TABLE = (-1.94, -2.80)   # just past the corridor, at the delivery table

# Static scene features in world coordinates.
SHELF_X_CENTERS = [-1.735, -0.85, 0.035, 0.92, 1.805]      # shelf row A..E (x), at y=3.323
SHELF_Y_BACK = 3.323
SHELF_Y_FRONT = 3.243
PICK_Y_FAR, PICK_Y_NEAR = 3.25, 1.70                      # yellow picking zone (y)
DELIVERY_CENTER = (-1.94, -3.41)
DELIVERY_HALF_X, DELIVERY_HALF_Y = 0.48, 0.48             # enclosed yellow delivery zone
START_CENTER = (1.92, -3.17)                             # robot starting area
BOX_EAST_X = 0.53                                        # corridor east wall (world x)
BOX_WEST_X = -2.5                                        # corridor west wall
BOX_CORRIDOR_Y_LO = WORLD_Y_MIN                          # box corridor spans y in [-3.75, 1.70]
BOX_CORRIDOR_Y_HI = PICK_Y_NEAR


def world_to_map(wx: float, wy: float) -> tuple[float, float]:
    px = MAP_X0 + (wx - WORLD_X_MIN) * (MAP_X1 - MAP_X0) / (WORLD_X_MAX - WORLD_X_MIN)
    py = MAP_Y1 - (wy - WORLD_Y_MIN) * (MAP_Y1 - MAP_Y0) / (WORLD_Y_MAX - WORLD_Y_MIN)
    return px, py


def map_to_world(px: float, py: float) -> tuple[float, float]:
    wx = WORLD_X_MIN + (px - MAP_X0) * (WORLD_X_MAX - WORLD_X_MIN) / (MAP_X1 - MAP_X0)
    wy = WORLD_Y_MIN + (MAP_Y1 - py) * (WORLD_Y_MAX - WORLD_Y_MIN) / (MAP_Y1 - MAP_Y0)
    return wx, wy


@dataclass
class Stream:
    title: str
    image: Optional[np.ndarray] = None
    pending_message: Optional[Image] = None
    arrivals: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    error: str = ""
    last_image_update: float = 0.0

    def observe(self) -> bool:
        now = time.monotonic()
        self.arrivals.append(now)
        if now - self.last_image_update < 1.0 / DISPLAY_FPS:
            return False
        self.last_image_update = now
        return True

    def push(self, image: np.ndarray) -> None:
        self.image = image
        self.error = ""

    @property
    def fps(self) -> float:
        if len(self.arrivals) < 2:
            return 0.0
        span = self.arrivals[-1] - self.arrivals[0]
        return (len(self.arrivals) - 1) / span if span > 0 else 0.0


class ImageReceiver(Node):
    """Receive image traffic independently from telemetry and services."""

    def __init__(self, streams: dict[str, Stream]) -> None:
        super().__init__("supermarket_dashboard_images")
        self.streams = streams
        topics = {
            "head": "/head_camera/color/image_raw",
            "detect": "/kele/result_image",
            "left": "/left_camera/color/image_raw",
            "right": "/right_camera/color/image_raw",
        }
        self._subscriptions = []
        self._callback_groups = []
        for key, topic in topics.items():
            callback_group = MutuallyExclusiveCallbackGroup()
            self._callback_groups.append(callback_group)
            self._subscriptions.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, name=key: self.on_image(name, msg),
                    IMAGE_QOS,
                    callback_group=callback_group,
                )
            )

    def on_image(self, name: str, msg: Image) -> None:
        stream = self.streams[name]
        stream.arrivals.append(time.monotonic())
        stream.pending_message = msg


class Dashboard(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_visual_dashboard")
        self.bridge = CvBridge()
        self.streams = {
            "head": Stream("Head RGB"),
            "detect": Stream("YOLO /kele/result_image"),
            "left": Stream("Left wrist RGB"),
            "right": Stream("Right wrist RGB"),
        }
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan", self.on_scan, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Detection3DArray, "/kele/detections", self.on_detection, 10)
        self.create_subscription(String, "/referee/gameinfo", self.on_gameinfo, 10)
        self.create_subscription(Int32, "/referee/score", self.on_score, 10)
        self.create_subscription(
            String, "/supermarket_sorting/runtime", self.on_runtime, 10
        )
        self.control_publisher = self.create_publisher(
            String, "/supermarket_sorting/control", 5
        )
        self.reset_client = self.create_client(Trigger, "/supermarket_sorting/reset_run")
        self.client_state_reset = self.create_client(Trigger, "/supermarket_sorting/reset_client")
        self.baseline_restart = self.create_client(
            Trigger, "/supermarket_sorting/restart_baseline"
        )

        self.base = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.scan_points: list[tuple[float, float]] = []
        self.scan_fps = 0.0
        self._scan_times: deque[float] = deque(maxlen=60)
        self.goal: Optional[tuple[float, float]] = None
        self.goal_frame = "--"
        self.goal_publisher = self.create_publisher(PoseStamped, "/nav/goal", 5)
        self.joints: dict[str, float] = {}
        self.joint_arrivals: deque[float] = deque(maxlen=100)
        self.target: Optional[tuple[float, float, float, float, str]] = None
        self.gameinfo = "waiting for /referee/gameinfo"
        self.score = 0
        self.gpu = "GPU: collecting..."
        self.last_gpu_poll = 0.0
        self.started = time.monotonic()
        self.reset_pending = False
        self.reset_status = "Reset: ready"
        self.runtime: dict[str, object] = {}
        self.runtime_updated = 0.0
        self.boot_id: Optional[str] = None
        self.pending_seed = 0
        self.pending_randomize = True
        self.pending_gs = False
        self.scene_dirty = False
        self.scene_apply_pending = False

    def decode_pending_images(self) -> None:
        """Decode only each stream's newest frame in the GUI thread."""
        for name, stream in self.streams.items():
            msg, stream.pending_message = stream.pending_message, None
            if msg is None:
                continue
            try:
                first_image = stream.image is None
                stream.push(self.bridge.imgmsg_to_cv2(msg, "bgr8"))
                if first_image:
                    self.get_logger().info(
                        f"first {name} image: {msg.width}x{msg.height} {msg.encoding}"
                    )
            except Exception as exc:  # Keep the dashboard alive if one stream is malformed.
                stream.error = str(exc)
                self.get_logger().warning(f"{name} image conversion failed: {exc}")

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.base = (p.x, p.y, yaw)
        self.velocity = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def on_scan(self, msg: LaserScan) -> None:
        now = time.monotonic()
        self._scan_times.append(now)
        x, y, yaw = self.base
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        points: list[tuple[float, float]] = []
        n = len(msg.ranges)
        stride = max(1, n // 120)
        for index in range(0, n, stride):
            r = msg.ranges[index]
            if not math.isfinite(r) or r <= 0.1 or r > GOAL_MAX_RANGE:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            local_x = r * math.cos(angle)
            local_y = r * math.sin(angle)
            wx = x + local_x * cos_y - local_y * sin_y
            wy = y + local_x * sin_y + local_y * cos_y
            points.append((wx, wy))
        self.scan_points = points
        if len(self._scan_times) >= 2:
            span = self._scan_times[-1] - self._scan_times[0]
            self.scan_fps = (len(self._scan_times) - 1) / span if span > 0 else 0.0

    def set_nav_goal(self, wx: float, wy: float, frame: str = "map") -> None:
        self.goal = (wx, wy)
        self.goal_frame = frame
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(wx)
        msg.pose.position.y = float(wy)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_publisher.publish(msg)
        self.reset_status = f"Nav goal ({frame}): x={wx:+.3f} y={wy:+.3f}"

    def on_joints(self, msg: JointState) -> None:
        self.joints = dict(zip(msg.name, msg.position))
        self.joint_arrivals.append(time.monotonic())

    def on_detection(self, msg: Detection3DArray) -> None:
        if not msg.detections:
            return
        detection = msg.detections[0]
        score = 0.0
        if detection.results:
            result = detection.results[0]
            position = result.pose.pose.position
            score = float(result.hypothesis.score)
        else:
            center = detection.bbox.center
            position = getattr(center, "position", center)
        self.target = (position.x, position.y, position.z, score, msg.header.frame_id or "?")

    def on_gameinfo(self, msg: String) -> None:
        self.gameinfo = msg.data

    def on_score(self, msg: Int32) -> None:
        self.score = msg.data

    def on_runtime(self, msg: String) -> None:
        try:
            runtime = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        new_boot_id = str(runtime.get("boot_id", ""))
        restarted = self.boot_id is not None and new_boot_id != self.boot_id
        self.boot_id = new_boot_id
        self.runtime = runtime
        self.runtime_updated = time.monotonic()
        if not self.scene_dirty or restarted:
            self.pending_seed = int(runtime.get("seed", 0))
            self.pending_randomize = bool(runtime.get("randomize", True))
            self.pending_gs = bool(runtime.get("use_gs", False))
            self.scene_dirty = False
        if restarted and self.scene_apply_pending:
            self.scene_apply_pending = False
            self.restart_baseline_after_scene("Scene rebuilt")

    def publish_control(self, payload: dict[str, object]) -> None:
        self.control_publisher.publish(
            String(data=json.dumps(payload, separators=(",", ":")))
        )

    def set_rate(self, rate: float) -> None:
        self.publish_control({"action": "set_rate", "target_rtf": rate})
        self.reset_status = "Simulation rate: MAX" if rate == 0 else f"Simulation rate: {rate:.1f}x"

    def change_seed(self, delta: int) -> None:
        self.pending_seed += delta
        self.scene_dirty = True
        self.reset_status = "Seed changed; press APPLY SCENE"

    def toggle_gs(self) -> None:
        self.pending_gs = not self.pending_gs
        self.scene_dirty = True
        self.reset_status = "3DGS changed; press APPLY SCENE"

    def toggle_randomize(self) -> None:
        self.pending_randomize = not self.pending_randomize
        self.scene_dirty = True
        self.reset_status = "Random layout changed; press APPLY SCENE"

    def apply_scene(self) -> None:
        if self.scene_apply_pending:
            return
        self.scene_apply_pending = True
        self.reset_status = "Rebuilding scene (server stays in container)..."
        self.publish_control(
            {
                "action": "restart_scene",
                "seed": self.pending_seed,
                "randomize": self.pending_randomize,
                "use_gs": self.pending_gs,
            }
        )

    def on_scene_client_reset_done(self, future) -> None:
        try:
            response = future.result()
            self.reset_status = (
                "Scene applied + client reset"
                if response.success
                else f"Client reset failed: {response.message}"[:46]
            )
            if response.success:
                self.target = None
        except Exception as exc:
            self.reset_status = f"Client reset error: {exc}"[:46]

    def restart_baseline_after_scene(self, prefix: str) -> None:
        if self.baseline_restart.service_is_ready():
            self.reset_status = f"{prefix}; restarting baseline..."
            future = self.baseline_restart.call_async(Trigger.Request())
            future.add_done_callback(self.on_baseline_restart_done)
        elif self.client_state_reset.service_is_ready():
            self.reset_status = f"{prefix}; resetting client state..."
            future = self.client_state_reset.call_async(Trigger.Request())
            future.add_done_callback(self.on_scene_client_reset_done)
        else:
            self.reset_pending = False
            self.reset_status = f"{prefix}; baseline unavailable"

    def on_baseline_restart_done(self, future) -> None:
        self.reset_pending = False
        try:
            response = future.result()
            self.reset_status = (
                "Server reset + baseline restarted"
                if response.success
                else f"Baseline restart failed: {response.message}"[:46]
            )
            if response.success:
                self.target = None
        except Exception as exc:
            self.reset_status = f"Baseline restart error: {exc}"[:46]

    def request_reset(self) -> None:
        if self.reset_pending:
            return
        if not self.reset_client.service_is_ready():
            self.reset_status = "Server reset unavailable"
            return
        if not (
            self.baseline_restart.service_is_ready()
            or self.client_state_reset.service_is_ready()
        ):
            self.reset_status = "Baseline control unavailable"
            return
        self.reset_pending = True
        self.reset_status = "Resetting server..."
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(self.on_server_reset_done)

    def on_server_reset_done(self, future) -> None:
        try:
            response = future.result()
            if response.success:
                self.restart_baseline_after_scene("Scene reset")
            else:
                self.reset_pending = False
                self.reset_status = f"Reset failed: {response.message}"[:42]
        except Exception as exc:
            self.reset_pending = False
            self.reset_status = f"Reset error: {exc}"[:42]

    def on_client_reset_done(self, future) -> None:
        self.reset_pending = False
        try:
            response = future.result()
            if response.success:
                self.reset_status = "Reset OK: server + client"
                self.target = None
            else:
                self.reset_status = f"Client reset failed: {response.message}"[:42]
        except Exception as exc:
            self.reset_status = f"Client reset error: {exc}"[:42]

    def set_manual_drive(self, linear: float, angular: float) -> None:
        self.publish_control(
            {
                "action": "manual_drive",
                "enabled": True,
                "linear": linear,
                "angular": angular,
            }
        )
        self.reset_status = (
            "Manual STOP"
            if linear == 0.0 and angular == 0.0
            else f"Manual drive v={linear:+.2f} w={angular:+.2f}"
        )

    def release_manual_drive(self) -> None:
        self.publish_control(
            {"action": "manual_drive", "enabled": False, "linear": 0.0, "angular": 0.0}
        )
        self.reset_status = "AUTO: baseline controls robot"

    @property
    def joint_fps(self) -> float:
        if len(self.joint_arrivals) < 2:
            return 0.0
        span = self.joint_arrivals[-1] - self.joint_arrivals[0]
        return (len(self.joint_arrivals) - 1) / span if span > 0 else 0.0

    def poll_gpu(self) -> None:
        now = time.monotonic()
        if now - self.last_gpu_poll < 1.0:
            return
        self.last_gpu_poll = now
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=1.0,
            ).strip()
            util, used, total, temp = [part.strip() for part in output.split(",")[:4]]
            self.gpu = f"GPU {util}% | VRAM {used}/{total} MiB | {temp} C"
        except Exception as exc:
            self.gpu = "GPU telemetry: use host nvidia-smi"


def fit_image(image: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    if image is None:
        cv2.putText(panel, "Waiting for image...", (28, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1, cv2.LINE_AA)
        return panel
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized = cv2.resize(image, (max(1, int(src_w * scale)), max(1, int(src_h * scale))))
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def draw_stream(canvas: np.ndarray, stream: Stream, x: int, y: int) -> None:
    tile = fit_image(stream.image, TILE_W, TILE_H)
    canvas[y : y + TILE_H, x : x + TILE_W] = tile
    cv2.rectangle(canvas, (x, y), (x + TILE_W, y + TILE_H), (70, 70, 70), 1)
    cv2.rectangle(canvas, (x, y), (x + TILE_W, y + 31), (16, 16, 16), -1)
    label = f"{stream.title}   {stream.fps:5.1f} FPS"
    cv2.putText(canvas, label, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (80, 230, 120), 1, cv2.LINE_AA)
    if stream.error:
        cv2.putText(canvas, stream.error[:60], (x + 10, y + TILE_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 255), 1, cv2.LINE_AA)


def telemetry_lines(node: Dashboard) -> tuple[list[str], list[str]]:
    x, y, yaw = node.base
    linear, angular = node.velocity
    slide = node.joints.get("slide_joint", float("nan"))
    head_yaw = node.joints.get("head_yaw_joint", float("nan"))
    head_pitch = node.joints.get("head_pitch_joint", float("nan"))
    left_grip = node.joints.get("left_arm_eef_gripper_joint", float("nan"))
    right_grip = node.joints.get("right_arm_eef_gripper_joint", float("nan"))
    if node.target is None:
        target = "Target: waiting for detection"
        confidence = "Detection: --"
    else:
        tx, ty, tz, score, frame = node.target
        target = f"Target ({frame}): x={tx:+.3f} y={ty:+.3f} z={tz:+.3f}"
        confidence = f"YOLO confidence: {score:.3f}"
    left = [
        f"Base world: x={x:+.3f}  y={y:+.3f}  yaw={math.degrees(yaw):+.1f} deg",
        f"Velocity: linear={linear:+.3f} m/s  angular={angular:+.3f} rad/s",
        target,
        confidence,
        f"Slide={slide:+.3f} m  Head yaw/pitch={head_yaw:+.3f}/{head_pitch:+.3f}",
        f"Grippers: left={left_grip:+.3f}  right={right_grip:+.3f}",
        f"Scan {node.scan_fps:.1f} Hz  pts={len(node.scan_points)}",
        "Landmarks: shelf_x=[-1.74,-0.85,0.04,0.92,1.81]y=3.32",
        "  table=(-1.94,-3.41)  start=(1.92,-3.17)  corridor x in [-2.47,0.53]",
    ]
    if node.goal is None:
        goal_line = "Nav goal: none (click arena map / GOAL buttons)"
    else:
        dx = node.goal[0] - x
        dy = node.goal[1] - y
        goal_line = (
            f"Nav goal ({node.goal_frame}): ({node.goal[0]:+.3f}, {node.goal[1]:+.3f}) "
            f"dist {math.hypot(dx, dy):.2f}m"
        )
    left.append(goal_line)
    runtime = node.runtime
    actual_rtf = float(runtime.get("actual_rtf", 0.0))
    total_rtf = float(runtime.get("total_rtf", 0.0))
    target_rtf = float(runtime.get("target_rtf", 0.0))
    target_label = "MAX" if target_rtf == 0 else f"{target_rtf:.2f}x"
    sim_time = float(runtime.get("sim_time", 0.0))
    wall_time = float(runtime.get("wall_time", 0.0))
    seed = runtime.get("seed", "--")
    randomize = "ON" if runtime.get("randomize", False) else "OFF"
    use_gs = "ON" if runtime.get("use_gs", False) else "OFF"
    renderer = runtime.get("renderer", "waiting")
    gl_backend = runtime.get("gl_backend", "--")
    display_mode = "headless" if runtime.get("headless", False) else "windowed"
    integrator = runtime.get("integrator", "--")
    timestep_ms = float(runtime.get("timestep", 0.0)) * 1000.0
    sensor_hz = float(runtime.get("sensor_hz", 0.0))
    wrist_hz = float(runtime.get("wrist_render_hz", sensor_hz))
    gs_points = int(runtime.get("gs_total_points", 0))
    right = [
        f"Score {node.score}  |  Joint state {node.joint_fps:.1f} Hz",
        f"Head {node.streams['head'].fps:.1f} FPS | YOLO {node.streams['detect'].fps:.1f} FPS | wrist render {wrist_hz:.0f} Hz",
        f"Sim {sim_time:7.2f}s  |  Wall {wall_time:7.2f}s",
        f"RTF actual {actual_rtf:.3f}x  |  total {total_rtf:.3f}x  |  target {target_label}",
        f"Scene seed {seed}  |  randomize {randomize}",
        f"3DGS {use_gs}  |  {renderer}  |  GL {gl_backend}/{display_mode}",
        f"Physics {integrator} | dt {timestep_ms:.2f} ms | GS {gs_points / 1e6:.2f}M pts",
        node.gpu,
    ]
    return left, right


def draw_button(
    canvas: np.ndarray,
    bounds: tuple[int, int, int, int],
    label: str,
    active: bool = False,
    warning: bool = False,
) -> None:
    x0, y0, x1, y1 = bounds
    if warning:
        fill, border = (45, 85, 130), (90, 170, 250)
    elif active:
        fill, border = (55, 105, 65), (90, 230, 125)
    else:
        fill, border = (58, 58, 58), (105, 105, 105)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
    tx = x0 + max(4, (x1 - x0 - text_size[0]) // 2)
    ty = y0 + (y1 - y0 + text_size[1]) // 2
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)


def draw_map(canvas: np.ndarray, node: Dashboard) -> None:
    x0, y0, x1, y1 = MAP_PANEL
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (34, 34, 40), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 85), 1)
    cv2.putText(canvas, "ARENA MAP", (x0 + 8, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"scan {node.scan_fps:.0f}Hz", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 140, 170), 1, cv2.LINE_AA)

    floor = (28, 34, 30)
    cv2.rectangle(canvas, (int(MAP_X0), int(MAP_Y0)), (int(MAP_X1), int(MAP_Y1)), floor, -1)
    cv2.rectangle(canvas, (int(MAP_X0), int(MAP_Y0)), (int(MAP_X1), int(MAP_Y1)), (120, 120, 130), 1)


    def rect(w0: float, y0w: float, w1: float, y1w: float, color, fill=-1, thickness: int = 1) -> None:
        px0, py0 = world_to_map(w0, y0w)
        px1, py1 = world_to_map(w1, y1w)
        a = (int(px0), int(py0))
        b = (int(px1), int(py1))
        cv2.rectangle(canvas, a, b, color, fill)
        if fill != -1:
            cv2.rectangle(canvas, a, b, color, thickness)

    # Corridor (box zone) shading.
    rect(WORLD_X_MIN, BOX_CORRIDOR_Y_LO, WORLD_X_MAX, BOX_CORRIDOR_Y_HI, (42, 36, 26), -1)

    def line(wx0: float, wy0: float, wx1: float, wy1: float, color, thickness: int = 1) -> None:
        px0, py0 = world_to_map(wx0, wy0)
        px1, py1 = world_to_map(wx1, wy1)
        cv2.line(canvas, (int(px0), int(py0)), (int(px1), int(py1)), color, thickness)

    line(BOX_EAST_X, BOX_CORRIDOR_Y_LO, BOX_EAST_X, BOX_CORRIDOR_Y_HI, (70, 160, 120))
    line(BOX_WEST_X, BOX_CORRIDOR_Y_LO, BOX_WEST_X, BOX_CORRIDOR_Y_HI, (70, 160, 120))

    # Yellow picking zone.
    rect(WORLD_X_MIN, PICK_Y_NEAR, WORLD_X_MAX, PICK_Y_FAR, (0, 150, 200), -1)
    # Start and delivery zones.
    for cx, cy in (START_CENTER, DELIVERY_CENTER):
        rect(cx - 0.4, cy - 0.4, cx + 0.4, cy + 0.4, (0, 200, 255), -1)

    # Shelf row (front face towards +y where products/kele sit).
    for sx in SHELF_X_CENTERS:
        rect(sx - 0.43, SHELF_Y_FRONT, sx + 0.43, SHELF_Y_BACK, (60, 60, 160), -1)

    # World-coordinate grid + axis labels (every 1.0 m in the /odom world frame).
    grid_col = (52, 60, 66)
    for gx in range(int(math.ceil(WORLD_X_MIN)), int(math.floor(WORLD_X_MAX)) + 1):
        px, _ = world_to_map(gx, 0.0)
        cv2.line(canvas, (int(px), int(MAP_Y0)), (int(px), int(MAP_Y1)), grid_col, 1)
        cv2.putText(canvas, f"x={gx}", (int(px) - 12, int(MAP_Y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 140, 150), 1, cv2.LINE_AA)
    for gy in range(int(math.ceil(WORLD_Y_MIN)), int(math.floor(WORLD_Y_MAX)) + 1):
        _, py = world_to_map(0.0, gy)
        cv2.line(canvas, (int(MAP_X0), int(py)), (int(MAP_X1), int(py)), grid_col, 1)
        cv2.putText(canvas, f"y={gy}", (int(MAP_X0) + 2, int(py) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 140, 150), 1, cv2.LINE_AA)

    # Laser obstacle overlay (live).
    for wx, wy in node.scan_points:
        px, py = world_to_map(wx, wy)
        cv2.circle(canvas, (int(px), int(py)), 1, (30, 90, 255), -1, cv2.LINE_AA)

    # Goal marker.
    if node.goal is not None:
        gx, gy = world_to_map(*node.goal)
        gx, gy = int(gx), int(gy)
        cv2.circle(canvas, (gx, gy), 7, (80, 230, 120), 2, cv2.LINE_AA)
        cv2.line(canvas, (gx - 10, gy), (gx + 10, gy), (80, 230, 120), 1)
        cv2.line(canvas, (gx, gy - 10), (gx, gy + 10), (80, 230, 120), 1)

    # Robot pose triangle (world +x -> right on map, +y -> up).
    bx, by, yaw = node.base
    rx, ry = world_to_map(bx, by)
    rx, ry = int(rx), int(ry)
    length = 16
    tip = (rx + int(length * math.cos(yaw)), ry - int(length * math.sin(yaw)))
    back_x, back_y = rx - int(length * 0.6 * math.cos(yaw)), ry + int(length * 0.6 * math.sin(yaw))
    left = (back_x + int(7 * math.cos(yaw + math.pi / 2)), back_y - int(7 * math.sin(yaw + math.pi / 2)))
    right = (back_x - int(7 * math.cos(yaw + math.pi / 2)), back_y + int(7 * math.sin(yaw + math.pi / 2)))
    tri = np.array([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(canvas, tri, (255, 70, 70))
    cv2.circle(canvas, (rx, ry), 3, (255, 70, 70), -1)

    # Goal preset buttons.
    draw_button(canvas, GOAL_SHELF_BUTTON, "GOAL SHELF")
    draw_button(canvas, GOAL_TABLE_BUTTON, "GOAL TABLE")
    goal_info = "click map to set nav goal"
    if node.goal is not None:
        dx = node.goal[0] - bx
        dy = node.goal[1] - by
        goal_info = f"goal ({node.goal[0]:+.2f},{node.goal[1]:+.2f}) dist {math.hypot(dx, dy):.2f}m"
    cv2.putText(canvas, goal_info, (x0 + 8, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 200, 160), 1, cv2.LINE_AA)


def render(node: Dashboard) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 24, dtype=np.uint8)
    positions = {
        "head": (12, 12),
        "detect": (322, 12),
        "left": (12, 222),
        "right": (322, 222),
    }
    for name, (x, y) in positions.items():
        draw_stream(canvas, node.streams[name], x, y)
    draw_map(canvas, node)

    cv2.rectangle(canvas, (12, 636), (948, 968), (42, 42, 42), -1)
    cv2.putText(canvas, "MANUAL DRIVE", (52, 671), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "SIM RATE", (260, 674), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
    current_rate = float(node.runtime.get("target_rtf", -1.0))
    for rate, bounds in RATE_BUTTONS.items():
        label = "MAX" if rate == 0 else f"{rate:g}x"
        draw_button(canvas, bounds, label, active=abs(current_rate - rate) < 1e-6)
    draw_button(
        canvas,
        RESET_BUTTON,
        "RESETTING..." if node.reset_pending else "RESET SCENE",
        warning=True,
    )
    game = node.gameinfo.replace("\n", " ")
    cv2.putText(canvas, game[:70], (260, 768), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"SEED {node.pending_seed}", (260, 728), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    draw_button(canvas, SEED_MINUS_BUTTON, "-")
    draw_button(canvas, SEED_PLUS_BUTTON, "+")
    draw_button(
        canvas,
        RANDOM_BUTTON,
        "RANDOM ON" if node.pending_randomize else "RANDOM OFF",
        active=node.pending_randomize,
    )
    draw_button(
        canvas,
        GS_BUTTON,
        "3DGS ON" if node.pending_gs else "3DGS OFF",
        active=node.pending_gs,
    )
    draw_button(canvas, APPLY_BUTTON, "APPLY SCENE", warning=node.scene_dirty)
    cv2.putText(canvas, node.reset_status[:25], (765, 726), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 190, 245), 1, cv2.LINE_AA)
    manual_active = bool(node.runtime.get("manual_drive", False))
    manual_linear = float(node.runtime.get("manual_linear", 0.0))
    manual_angular = float(node.runtime.get("manual_angular", 0.0))
    draw_button(
        canvas,
        MOVE_BUTTONS["forward"],
        "FORWARD",
        active=manual_active and manual_linear > 0.0,
    )
    draw_button(
        canvas,
        MOVE_BUTTONS["left"],
        "LEFT",
        active=manual_active and manual_angular > 0.0,
    )
    draw_button(
        canvas,
        MOVE_BUTTONS["stop"],
        "STOP",
        active=manual_active and manual_linear == 0.0 and manual_angular == 0.0,
        warning=True,
    )
    draw_button(
        canvas,
        MOVE_BUTTONS["right"],
        "RIGHT",
        active=manual_active and manual_angular < 0.0,
    )
    draw_button(
        canvas,
        MOVE_BUTTONS["back"],
        "BACK",
        active=manual_active and manual_linear < 0.0,
    )
    draw_button(canvas, MOVE_BUTTONS["auto"], "AUTO", active=not manual_active)
    cv2.putText(canvas, "WASD / Space / M", (22, 812), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (135, 135, 135), 1, cv2.LINE_AA)
    left, right = telemetry_lines(node)
    for index, line in enumerate(left):
        cv2.putText(canvas, line, (28, 820 + index * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 215, 215), 1, cv2.LINE_AA)
    for index, line in enumerate(right):
        if index == 0:
            color = (90, 230, 130)
        elif index == 3:
            actual = float(node.runtime.get("actual_rtf", 0.0))
            target = float(node.runtime.get("target_rtf", 0.0))
            if target == 0:
                color = (80, 210, 245)
            elif actual < target * 0.9:
                color = (80, 100, 255)
            else:
                color = (90, 230, 130)
        else:
            color = (215, 215, 215)
        cv2.putText(canvas, line, (492, 795 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Seed / 3DGS apply by rebuilding the scene; Q / Esc closes dashboard only", (492, 963), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 130, 130), 1, cv2.LINE_AA)
    return canvas


def mouse_callback(event: int, x: int, y: int, _flags: int, node: Dashboard) -> None:
    if event != cv2.EVENT_LBUTTONUP:
        return
    for bounds, wx, wy, frame in (
        (GOAL_SHELF_BUTTON, *GOAL_SHELF, "shelf"),
        (GOAL_TABLE_BUTTON, *GOAL_TABLE, "table"),
    ):
        x0, y0, x1, y1 = bounds
        if x0 <= x <= x1 and y0 <= y <= y1:
            node.set_nav_goal(wx, wy, frame)
            return
    mx0, my0, mx1, my1 = MAP_PANEL
    if mx0 <= x <= mx1 and my0 <= y <= my1:
        wx, wy = map_to_world(x, y)
        node.set_nav_goal(wx, wy)
        return
    x0, y0, x1, y1 = RESET_BUTTON
    if x0 <= x <= x1 and y0 <= y <= y1:
        node.request_reset()
        return
    for rate, (x0, y0, x1, y1) in RATE_BUTTONS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            node.set_rate(rate)
            return
    for bounds, action in (
        (SEED_MINUS_BUTTON, lambda: node.change_seed(-1)),
        (SEED_PLUS_BUTTON, lambda: node.change_seed(1)),
        (RANDOM_BUTTON, node.toggle_randomize),
        (GS_BUTTON, node.toggle_gs),
        (APPLY_BUTTON, node.apply_scene),
    ):
        x0, y0, x1, y1 = bounds
        if x0 <= x <= x1 and y0 <= y <= y1:
            action()
            return
    drive_actions = {
        "forward": lambda: node.set_manual_drive(0.25, 0.0),
        "back": lambda: node.set_manual_drive(-0.20, 0.0),
        "left": lambda: node.set_manual_drive(0.0, 0.55),
        "right": lambda: node.set_manual_drive(0.0, -0.55),
        "stop": lambda: node.set_manual_drive(0.0, 0.0),
        "auto": node.release_manual_drive,
    }
    for name, bounds in MOVE_BUTTONS.items():
        x0, y0, x1, y1 = bounds
        if x0 <= x <= x1 and y0 <= y <= y1:
            drive_actions[name]()
            return


def main() -> None:
    rclpy.init()
    node = Dashboard()
    image_node = ImageReceiver(node.streams)
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(image_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)
    cv2.moveWindow(WINDOW, 950, 35)
    cv2.setMouseCallback(WINDOW, mouse_callback, node)
    preview_path = os.getenv("SUPERMARKET_DASHBOARD_PREVIEW", "")
    preview_interval = 1.0 / DISPLAY_FPS
    last_preview = 0.0
    try:
        while rclpy.ok():
            frame_started = time.monotonic()
            node.poll_gpu()
            node.decode_pending_images()
            frame = render(node)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if preview_path and time.monotonic() - last_preview >= preview_interval:
                last_preview = time.monotonic()
                try:
                    cv2.imwrite(preview_path, frame)
                except Exception as exc:  # never let a preview write kill the dashboard
                    print(f"[dashboard] preview write failed: {exc}", flush=True)
            if key in (27, ord("q"), ord("Q")):
                break
            key_actions = {
                ord("w"): lambda: node.set_manual_drive(0.25, 0.0),
                ord("s"): lambda: node.set_manual_drive(-0.20, 0.0),
                ord("a"): lambda: node.set_manual_drive(0.0, 0.55),
                ord("d"): lambda: node.set_manual_drive(0.0, -0.55),
                ord(" "): lambda: node.set_manual_drive(0.0, 0.0),
                ord("m"): node.release_manual_drive,
            }
            action = key_actions.get(key)
            if action is not None:
                action()
            remaining = 1.0 / DISPLAY_FPS - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        executor_thread.join(timeout=2.0)
        image_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
