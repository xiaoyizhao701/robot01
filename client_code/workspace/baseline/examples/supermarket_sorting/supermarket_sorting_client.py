#!/usr/bin/env python3
"""ROS2 client for the Supermarket Sorting Task.

Drive MMK2 to the target product's shelf slot, pick the visually detected
product (any of the 9 kinds), and place it on the delivery table. The client
keeps a 19-d target_control, computes right-arm joints with MMK2Kdl, and
publishes to the controller command topics exposed by the server.

The picked kind is selected with --kind (default kele). Slot location and
per-kind grasp parameters come from retail_competition_layout.json and
grasp_config.py, so a single state machine handles all 9 products.

Subscribes:
  /slamware_ros_sdk_server_node/odom   (nav_msgs/Odometry)  base pose in world
  /joint_states                        (sensor_msgs/JointState) 17 joints
  /kele/detections                    (vision_msgs/Detection3DArray, world) multi-class
Publishes:
  /cmd_vel, /spine.../commands, /head.../commands,
  /{left,right}_arm_forward_position_controller/commands
"""
import os
import json
import math
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Float64
from collections import deque

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray

from discoverse.utils import step_func
from mmk2_kdl import MMK2Kdl
from grasp_config import get_grasp
from perception.rack_map import RackMap

# ---- scene constants (world frame, +X east / +Y north). ----
TABLE_ORIGIN = np.array([-1.940, -3.410, 0.0])   # delivery_table; top surface z~0.77
YAW_NORTH = math.pi / 2.0
YAW_SOUTH = -math.pi / 2.0

YELLOW_MID_Y = 2.50           # shelf front lane (靠近货架, 便于抓取)
GRASP_YAW_EAST_BIAS_DEG = 11.0
GRASP_YAW = math.pi / 2.0 - math.radians(GRASP_YAW_EAST_BIAS_DEG)  # slightly east of north to square the gripper visually
# After the grasp, drive to the shelf LEFT-front (-1.52,2.30), then avoid the
# obstacle corridor down to the delivery table.
ROUTE_TO_TABLE = [[-1.52, 2.30], [-1.20, 2.30],
                  [-1.20, -0.70], [-1.20, -2.80], [-1.94, -3.10]]

# 动态走廊路径失败时的固定回退(seed 62 验证过的折线, 全程在走廊内)
FALLBACK_CORRIDOR = [
    [-1.00, 2.55], [-1.00, 1.80], [0.05, 0.60],
    [0.05, -1.20], [-1.10, -1.20], [-1.10, -2.20],
]

# ---- manipulation params that are kind-independent. ----
GRASP_ARM = "r"
HEAD_PITCH = -0.6
# Head pitch per shelf level so the head camera actually sees the product: a
# raised torso (L1 slide) needs a steeper look-down.
LEVEL_HEAD_PITCH = {"L1": -1.00, "L2": -0.70, "L3": -0.40}
SLIDE_PRE, SLIDE_LIFT = 0.11, -0.04
GRIP_OPEN = 1.0

# Slide-height anchor: the kele baseline (SLIDE_GRASP=0.11) targets L2 slots at
# world z ~0.9235. Other levels are reached by the same linear offset so the arm
# workspace stays over the slot. (Calibrate the offset if the body geometry shifts.)
KELE_Z_ANCHOR = 0.9235
KELE_SLIDE_ANCHOR = 0.11
SLIDE_Z_OFFSET = KELE_Z_ANCHOR - KELE_SLIDE_ANCHOR
# Spine slide needed for each shelf level so the arm reaches the slot with the
# natural grasp orientation (IK-verified): L1 low slot -> higher slide, L3 high
# slot -> low slide. The old z-based linear formula only worked for L2.
LEVEL_SLIDE = {"L1": 0.30, "L2": 0.15, "L3": 0.00}
# Rotation for placement: use R_ARM_NATURAL (top-down) so ALL kinds can reach the table
PLACE_ROTATION = np.array([
    [0.0, -1.0, 0.0],
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
])


# Open-space deploy pose and creep stop are offsets from the vision-estimated
# product center, not from any shelf slot coordinate. Defaults below are
# overwritten per-kind from grasp_config in __init__.
DEPLOY_OFFSET = np.array([-0.011, -0.220, -0.010])
CREEP_STOP_DY = 0.035
GRASP_ROT = np.eye(3)
GRIP_CLOSE = 0.08
REACH_Z_MIN, REACH_Z_MAX = 0.70, 1.15
SLIDE_GRASP = 0.11

# perception gating: the arm is posed only after vision gives a stable reachable point.
DETECT_DWELL = float(os.getenv("DETECT_DWELL", "1.0"))  # s to let head settle before locking
DETECT_MIN_SAMPLES = 5            # min detection frames to trust vision
# 抓取失败后的重试次数(开发/录像可设 0 省时间; 比赛默认 2)
GRASP_MAX_ATTEMPTS = int(os.getenv("GRASP_MAX_ATTEMPTS", "2"))
# 开发/录像: 跳过视觉搜索(配合 --prior-fallback 直接按先验走, 省 ~10s/单)
SEARCH_SKIP = os.getenv("SEARCH_SKIP", "0") == "1"
REACH_FWD_MIN, REACH_FWD_MAX = 0.35, 0.60   # R_ARM_SIDE 侧抓可达窗口(前向) — IK 到不了就别锁
REACH_LATERAL_MAX = 0.30         # m sideways: accept a product roughly ahead
                                 # from 0.14 so the fixed shelf-approach yaw still
                                 # admits the directly-ahead product; the arm IK then
                                 # does the final centering)
VISION_SURFACE_TO_CENTER_FWD = 0.0265   # product radius: RGB-D point is on the visible front surface
DEPLOY_CART_TOL = 0.090                   # m; friction leaves a small joint residual at the target
DEPLOY_JOINT_TOL = 0.200                  # rad; friction residual (static friction in the arm joints)
DEPLOY_ROT_TOL = 0.35                     # rad; wrist must be close to the grasp attitude
CREEP_SPEED = 0.06                                # m/s forward while creeping into the shelf
CREEP_YAW_KP = 4.0                                # hold heading firmly so the creep goes dead straight
LIFT_AMOUNT = 0.05                                # 澶逛綇鍚庣珫鐩存姮璧烽噺(鍑忓皬 slide),璁╃墿浣撶寮€闅旀澘鍐嶅€掕溅
# Placement: robot faces SOUTH at the table; arm must reach OUT over the table top
# (z~0.77) and set the object down. Offsets are world-frame (TABLE_ORIGIN z=0).
PLACE_LOWER_SLIDE = 0.17                          # 鏉剧埅鍓嶅皯闄嶄竴鐐?閬垮厤澶圭埅/鐡惰韩搴曢儴纰版

# JointState names (order documented by the server).
JOINT_NAMES = [
    "slide_joint", "head_yaw_joint", "head_pitch_joint",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_eef_gripper_joint",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
]
INIT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
INIT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

# top-level phases
NAV_SHELF, DEPLOY, CREEP, CLOSE, LIFT, RETREAT, NAV_TABLE, PLACE, DONE = range(9)
PHASE_NAME = {NAV_SHELF: "nav->shelf", DEPLOY: "deploy-arm", CREEP: "creep-in", CLOSE: "close",
              LIFT: "lift", RETREAT: "retreat", NAV_TABLE: "nav->table", PLACE: "place", DONE: "done"}
RETREAT_SPEED = 0.20          # 鍊掕溅閫熷害 m/s


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PickPlaceClient(Node):
    def __init__(self, kind="kele", layout_path=None, finish_after_retreat=False, skip_grasp=False,
                 gt_target=False, shelf=None, prior_fallback=False):
        super().__init__("supermarket_sorting_client")
        self.kind = kind
        self.shelf = shelf
        self.finish_after_retreat = finish_after_retreat
        self.skip_grasp = skip_grasp
        self.gt_target = gt_target
        self.prior_fallback = prior_fallback
        self.kdl = MMK2Kdl()   # unified FK + analytical-grade IK (matches MMK2FIK on this mjcf)

        # ---- resolve slot + per-kind grasp params from layout / grasp_config ----
        if layout_path is None:
            # this file lives in examples/supermarket_sorting/ alongside the layout
            layout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "retail_competition_layout.json")
        with open(layout_path, "r") as f:
            layout = json.load(f)
        slots = [s for s in layout if s.get("object_kind") == kind]
        if self.shelf:
            slots = [s for s in slots if s.get("shelf") == self.shelf]
        if not slots:
            raise RuntimeError(f"no slot for kind '{kind}' shelf='{self.shelf}' in layout")
        # pick the first slot of this kind (calibration uses a single known target)
        self.slot = slots[0]
        slot_pos = np.array(self.slot["world_position"], dtype=float)
        self.slot_world = slot_pos

        # ---- 先验引导的搜索计划 (rack_map 只规划, 不控制机器人) ----
        self.rack = RackMap(layout_path)
        self.prior = self.rack.get_target_prior(kind, self.slot)
        self.search_plan = self.rack.search_plan(self.prior)
        self._plan_idx = 0
        self._plan_loops = 0
        self.search_level = self.prior["level"]
        self.get_logger().info(
            f"[prior] kind={kind} shelf={self.prior['shelf']} level={self.prior['level']} "
            f"region_x={self.prior['center'][0]:.2f} plan_steps={len(self.search_plan)}")

        g = get_grasp(kind)
        self.grip_close = float(g["grip_close"])
        self.deploy_offset = np.array(g["deploy_offset"], dtype=float)
        self.creep_dy = float(g["creep_stop_dy"])
        self.reach_z_min = float(g["reach_z_min"])
        self.reach_z_max = float(g["reach_z_max"])
        self.grasp_rot = np.array(g["grasp_rot"], dtype=float)
        # slide height scales from the kele anchor so the arm workspace covers the slot
        self.slide_grasp = float(LEVEL_SLIDE.get(str(self.slot.get("level", "L2")), 0.15))
        self.head_pitch = float(LEVEL_HEAD_PITCH.get(str(self.slot.get("level", "L2")), HEAD_PITCH))
        self.approach_x = float(slot_pos[0])
        # Straight to the shelf RIGHT-front (1.52, 2.30) to search, then pull up
        # to ~0.5 m in front of the target slot (2.80) where the arm can reach
        # without the wheels touching the shelf.
        # GT 模式: 路由直接到商品前 0.60m(手臂 0.6m 可达, 前缘距货架仍 ~0.5m 安全)
        if self.gt_target:
            self.route_to_shelf = [[1.52, YELLOW_MID_Y],
                                   [self.approach_x, float(slot_pos[1]) - 0.60]]
        else:
            # 视觉模式也用商品前 0.60m 的停靠点: 原来停在 YELLOW_MID_Y+0.50=3.00,
            # 距货架面仅 0.32m, 机械臂够不到部署点(IK 失败)且手臂/头会顶到货架
            # 把滑轨卡死 -> deploy 卡住被调度器杀。
            self.route_to_shelf = [[1.52, YELLOW_MID_Y],
                                   [self.approach_x, float(slot_pos[1]) - 0.60]]
        self.get_logger().info(
            f"target kind={kind} slot={self.slot['shelf']}{self.slot['level']}{self.slot['column']} "
            f"pos={np.round(slot_pos,3)} slide_grasp={self.slide_grasp:.3f} "
            f"grip_close={self.grip_close} deploy_offset={np.round(self.deploy_offset,3)}")

        # target_control: [base_lin, base_ang, slide, head_yaw, head_pitch,
        #                  l_arm(6), l_grip, r_arm(6), r_grip]
        self.tc = np.zeros(19)
        self.tc[5:11] = INIT_ARM_L
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[18] = GRIP_OPEN

        # smoothed command actually published for slide/head/arms/grippers (idx 2..18):
        # step_func slews `action` toward `tc` so a new IK target never snaps (fixes 鐬Щ).
        # base velocity (idx 0,1) is NOT smoothed here -- it has its own accel ramp.
        self.action = self.tc.copy()
        self.joint_move_ratio = np.ones(19)
        self.tc_prev = self.tc.copy()
        self.joint_slew = 1.2            # rad/s (and grip-units/s) for the fastest joint

        # latest feedback
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None          # dict name->pos
        self.jvel = None

        # nav/phase state
        self.phase = NAV_SHELF
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.sub_idx = 0
        self.sub_entered = False
        self.deploy_set = False
        self.place_sub = 0
        self.state_t0 = self.now()
        self.arm_target_set = False

        # stuck-recovery state for follow_route
        self._stuck_prev_xy = None
        self._stuck_ticks = 0
        self._stuck_recovering = False
        self._stuck_recover_t0 = 0.0
        self._stuck_recover_phase = ""  # "reverse" or "rotate"
        self._stuck_count = 0

        # ---- perception: target is locked only from /kele/detections of our kind. ----
        self.OBJECT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self._search_step = 0
        self._search_t0 = 0.0
        self.det_buf = deque(maxlen=30)   # recent vision detections of the product directly ahead (world xyz)
        self.target_locked = False
        self.held_ok = None
        self.grasp_attempts = 0
        self.last_wait_log = 0.0
        self.last_deploy_wait_log = 0.0

        # gains
        self.pos_tol, self.turn_tol = float(os.getenv("POS_TOL", "0.15")), 0.05
        # Slower, steadier base motion: high speed + the extended arm made the
        # robot twitchy (and prone to tipping). Halve the linear/angular caps and
        # soften the acceleration ramp.
        self.max_lin, self.max_ang = float(os.getenv("MAX_LIN", "0.45")), 0.90
        # velocity ramping (acceleration limits) so /cmd_vel never jumps -> smooth motion
        self.rate_hz = 50.0
        self.dt = 1.0 / self.rate_hz
        self.max_lin_acc, self.max_ang_acc = 0.5, 1.6   # m/s^2, rad/s^2
        self.des_lin = self.des_ang = 0.0               # desired (from controller)
        self.cur_lin = self.cur_ang = 0.0               # ramped (actually published)
        self.des_lat = self.cur_lat = 0.0               # lateral (strafe) velocity

        # io
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.larm_pub = self.create_publisher(Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)
        self.grasp_target_pub = self.create_publisher(Float64, "/supermarket_sorting/grasp_target", 5)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Detection3DArray, "/kele/detections", self.det_cb, 10)
        self.create_subscription(Float64MultiArray, "/supermarket_sorting/grasp_status",
                                 self.grasp_status_cb, 10)
        self.grasp_status = None     # 服务端接触/提升/稳定状态(真值)
        self.grasp_ok_ticks = 0      # 连续满足成功条件的 tick 数

        self.timer = self.create_timer(self.dt, self.tick)
        self.last_log = 0.0
        self._tcount = 0
        self.get_logger().info("pick-place client up; waiting for odom + joint_states...")

    # ---- ros helpers ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        if not getattr(self, "_odom_dbg", False):
            self._odom_dbg = True
            self.get_logger().info("ODOM_CB fired")
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self.base_yaw = math.atan2(R[1, 0], R[0, 0])
        # 翻车保护: 底盘 z 轴偏离竖直的角度(正常 <0.2rad, 翻了 >1rad)
        self.base_tilt = float(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def js_cb(self, msg):
        if not getattr(self, "_js_dbg", False):
            self._js_dbg = True
            self.get_logger().info("JS_CB fired")
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.jvel = {n: msg.velocity[i] for i, n in enumerate(msg.name) if i < len(msg.velocity)}

    def grasp_status_cb(self, msg):
        """服务端抓取真值: [left, right, gripped, x, y, z, speed, tilt_deg]."""
        if len(msg.data) >= 8:
            self.grasp_status = np.array(msg.data[:8], dtype=float)

    def det_cb(self, msg):
        """Accumulate the product of our kind directly ahead of the parked base (world frame).
        Before lock (DEPLOY, !deploy_set) this seeds the grasp target; after lock it keeps
        tracking the held object through CREEP/CLOSE/LIFT/RETREAT for a hold-success check."""
        if self.base_xy is None:
            return
        if self.phase == DEPLOY and not self.deploy_set:
            pass
        elif self.target_locked and self.phase in (CREEP, CLOSE, LIFT, RETREAT):
            pass
        else:
            return
        best, best_lat = None, REACH_LATERAL_MAX
        for det in msg.detections:
            if not det.results:
                continue
            if det.results[0].hypothesis.class_id != self.kind:
                continue
            pos = det.results[0].pose.pose.position
            pw = np.array([pos.x, pos.y, pos.z])
            # 区域过滤: 目标层一致 + 货架横向范围内(允许商品偏移/跨格/斜放),
            # 而不是死死卡在格位中心 ±0.30m。层用"目标先验的层"(不是当前搜索视角的层),
            # 否则搜索扫到别的层时目标会被误滤掉。
            lv = self.prior["level"]
            if not self.rack.in_region(pw, self.prior, lv):
                continue
            fp = self.world_to_footprint(pw)   # fp[0]=forward (ahead), fp[1]=lateral (left+)
            fwd, lat = fp[0], abs(fp[1])
            if fwd < REACH_FWD_MIN or fwd > REACH_FWD_MAX:
                continue                       # wrong depth: floor, far shelf, etc.
            if pw[2] < self.reach_z_min or pw[2] > self.reach_z_max:
                continue
            if lat < best_lat:
                best_lat, best = lat, pw       # most centered product wins
        if best is not None:
            self.det_buf.append(best)

    def _vision_to_object_center(self, p_world):
        """Convert a visible RGB-D surface point into an estimated product center."""
        fp = self.world_to_footprint(p_world)
        fp[0] += VISION_SURFACE_TO_CENTER_FWD
        return self.footprint_to_world(fp)

    def _lock_target(self):
        """Lock a reachable target of our kind from recent vision detections."""
        if len(self.det_buf) < DETECT_MIN_SAMPLES:
            if self.now() - self.last_wait_log > 1.0:
                self.get_logger().info(
                    f"[perception] waiting for {self.kind} detections "
                    f"({len(self.det_buf)}/{DETECT_MIN_SAMPLES})")
                self.last_wait_log = self.now()
            return False

        arr = np.array(list(self.det_buf))
        candidate = np.median(arr, axis=0)
        fp = self.world_to_footprint(candidate)
        if (
            fp[0] < REACH_FWD_MIN or fp[0] > REACH_FWD_MAX
            or abs(fp[1]) > REACH_LATERAL_MAX
            or candidate[2] < self.reach_z_min or candidate[2] > self.reach_z_max
        ):
            self.get_logger().warn(
                f"[perception] discard unreachable {self.kind} candidate: "
                f"world={np.round(candidate,3)} fp={np.round(fp,3)}")
            self.det_buf.clear()
            return False

        self.OBJECT_WORLD = self._vision_to_object_center(candidate)
        self._send_grasp_target()
        # 深度/反投影有系统性偏深(实测 YOLO 位置比真值深 ~35mm), 补偿后再算目标。
        self.OBJECT_WORLD[1] -= 0.035
        self.DEPLOY_WORLD  = self.OBJECT_WORLD + self.deploy_offset
        # 部署目标在商品南 ~0.22m(官方参数, 手臂在货架外安全摆臂), 不再需要深度钳制。
        self.CREEP_STOP_Y  = self.OBJECT_WORLD[1] + self.creep_dy   # EE 爬进到商品南 creep_dy 处
        self.target_locked = True
        self.get_logger().info(
            f"[perception] {self.kind} locked from vision: "
            f"raw={np.round(candidate,3)} OBJECT={np.round(self.OBJECT_WORLD,3)}  "
            f"CREEP_STOP_Y={self.CREEP_STOP_Y:.4f}  samples={len(self.det_buf)}")
        return True

    def _corridor_route(self, from_table):
        """按当前种子的箱位动态计算走廊安全折线(自包含网格 A*)。

        走廊区域 x∈[-2.45,0.52], y∈[-3.0,2.6] 建独立网格(走廊外视为墙),
        把箱位(旋转矩形+膨胀)标为障碍, 用 8 邻域 A* 求南北通路并简化成
        ~0.9m 间隔路径点。不依赖共享 DWA 地图(它会被 lidar 噪声/墙端绕行
        影响)。换障碍物种子自动适配。
        """
        try:
            import obstacle_layout as _ol
            _seed = int(os.getenv("OBSTACLE_SEED", "62"))
            _g = _ol.generate_obstacle_layout(seed=_seed)
            _boxes = [(p[0] - 0.96, p[1] - 1.01, _g.yaws[k], 0.6, 0.4)
                      for k, p in _g.positions.items()]
            RES = 0.1
            XMIN, XMAX = -2.45, 0.52
            YMIN, YMAX = -3.0, 2.6
            xs = np.arange(XMIN, XMAX + 1e-9, RES)
            ys = np.arange(YMAX, YMIN - 1e-9, -RES)
            nx, ny = len(xs), len(ys)
            occ = np.zeros((ny, nx), bool)
            for (cx, cy, yaw, sx, sy) in _boxes:
                hx, hy = sx * 0.5 + 0.30, sy * 0.5 + 0.30
                c, s = math.cos(yaw), math.sin(yaw)
                for iy, y in enumerate(ys):
                    dy = y - cy
                    if abs(dy) > math.hypot(hx, hy):
                        continue
                    for ix, x in enumerate(xs):
                        dx = x - cx
                        lx = c * dx + s * dy
                        ly = -s * dx + c * dy
                        if abs(lx) <= hx and abs(ly) <= hy:
                            occ[iy, ix] = True
            # 8邻域 A* (1, √2 代价)
            import heapq
            def idx(x, y):
                return (int(round((y - YMAX) / -RES)), int(round((x - XMIN) / RES)))
            si, sj = idx(-1.0, 2.55)
            gi, gj = idx(-1.30, -2.55)
            if occ[si, sj] or occ[gi, gj]:
                return None
            open_set = [(0.0, si, sj)]
            came = {}
            gcost = {idx(-1.0, 2.55): 0.0}
            dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            dcost = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
            found = False
            while open_set:
                f, ci, cj = heapq.heappop(open_set)
                if (ci, cj) == (gi, gj):
                    found = True
                    break
                for (di, dj), dc in zip(dirs, dcost):
                    ni, nj = ci + di, cj + dj
                    if not (0 <= ni < ny and 0 <= nj < nx) or occ[ni, nj]:
                        continue
                    ng = gcost.get((ci, cj), 1e9) + dc
                    if ng < gcost.get((ni, nj), 1e9):
                        gcost[(ni, nj)] = ng
                        heapq.heappush(open_set, (ng + abs(ni - gi) + abs(nj - gj), ni, nj))
                        came[(ni, nj)] = (ci, cj)
            if not found:
                return None
            # 回溯路径
            pts = []
            cur = (gi, gj)
            while cur in came:
                pts.append((xs[cur[1]], ys[cur[0]]))
                cur = came[cur]
            pts.append((xs[sj], ys[si]))
            pts.reverse()
            # 简化: 每 ~0.9m 取一点
            out = [list(pts[0])]
            acc = 0.0
            prev = pts[0]
            for p in pts[1:]:
                acc += math.hypot(p[0] - prev[0], p[1] - prev[1])
                prev = p
                if acc >= 0.9:
                    out.append([p[0], p[1]])
                    acc = 0.0
            if out[-1] != list(pts[-1]):
                out.append(list(pts[-1]))
            self.get_logger().info(
                f"[nav] corridor route computed: {len(out)} pts seed={_seed}")
            return out
        except Exception as _e:
            self.get_logger().warn(f"[nav] corridor route failed: {_e}")
            return None

    def _lock_from_prior(self):
        """视觉搜索失败时的兜底: 直接用布局 JSON 里的格位世界坐标锁定目标。

        仅用于开发/录像(视觉域与训练数据不一致时保证流程完整), 比赛正式运行
        不应使用(规则 12 禁止读取布局真值)。"""
        self.OBJECT_WORLD = self.slot_world.copy()
        self._send_grasp_target()
        self.DEPLOY_WORLD = self.OBJECT_WORLD + self.deploy_offset
        self.CREEP_STOP_Y = float(self.OBJECT_WORLD[1]) + self.creep_dy
        self.target_locked = True
        self.get_logger().warn(
            f"[fallback] vision search failed; locking from layout prior "
            f"OBJECT={np.round(self.OBJECT_WORLD,3)} DEPLOY={np.round(self.DEPLOY_WORLD,3)}")

    def _view_search(self):
        """Bring an unseen target into view.

        When the target kind is not detected from the parked pose, actively
        adjust the viewpoint (like a human leaning/raising the head): first
        strafe the base laterally to line up with the target slot, then
        micro-sweep the spine height and head pitch. Returns True while still
        searching, False when the caller should try to lock again.

        The current step's action is applied on EVERY tick (the DEPLOY phase
        resets slide/head each tick, so a one-shot set would be overwritten).
        """
        SEARCH_DWELL = 1.2      # s per search view
        SEARCH_MAX_LOOPS = int(os.getenv("SEARCH_LOOPS", "2"))  # bounded passes then fail
        SEARCH_Y = 2.45         # back up to this lane distance: close poses (2.8)
                                # see only the shelf frame / a too-steep L1 pitch
        if getattr(self, "_search_failed", False):
            return False        # plan exhausted: stop moving, only re-lock
        if self.base_xy[1] > SEARCH_Y + 0.05:
            # too close: back up until the whole shelf row is in view
            self.set_twist(-0.18, CREEP_YAW_KP * wrap_to_pi(GRASP_YAW - self.base_yaw))
            if self._tcount % 25 == 0:
                self.get_logger().info(f"[search] back up y={self.base_xy[1]:.2f} -> {SEARCH_Y}")
            return True
        n = len(self.search_plan)
        step = self.search_plan[self._plan_idx % n]
        self.search_level = step["level"]
        if self.now() - self._search_t0 >= SEARCH_DWELL:
            self._search_t0 = self.now()
            self._plan_idx += 1
            if self._plan_idx % n == 0:
                self._plan_loops += 1
                if self._plan_loops >= SEARCH_MAX_LOOPS:
                    self._search_failed = True
                    self.get_logger().warn(
                        f"[search] SEARCH_FAILED kind={self.kind} "
                        f"attempts={self._plan_idx} loops={self._plan_loops} "
                        f"reason=NOT_FOUND_AFTER_PLAN")
                    return False
        # strafe toward this view's lateral position (base +y/left = -x world)
        tx = self.approach_x + step["lateral"]
        x_err = tx - self.base_xy[0]
        vy = float(np.clip(-x_err * 0.8, -0.12, 0.12))
        if abs(x_err) < 0.04:
            vy = 0.0
        self.set_twist(0.0, CREEP_YAW_KP * wrap_to_pi(GRASP_YAW - self.base_yaw), vy)
        # apply the level's verified viewing pose
        self.tc[2] = step["slide"]
        self.tc[4] = step["pitch"]
        if self._tcount % 25 == 0:
            self.get_logger().info(
                f"[search] attempt={step['attempt']}/{n} level={step['level']} "
                f"lateral={step['lateral']:+.2f} slide={step['slide']:.2f} "
                f"pitch={step['pitch']:.2f} x_err={x_err:+.2f}")
        return True

    @property
    def slide_meas(self):
        return self.jpos.get("slide_joint", self.tc[2])

    @property
    def rarm_meas(self):
        return np.array([self.jpos.get(f"right_arm_joint{i+1}", self.tc[12 + i]) for i in range(6)])

    # ---- frames ----
    def world_to_footprint(self, p_world):
        d = np.array(p_world, dtype=float) - np.array([self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def footprint_to_world(self, fp):
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([self.base_xy[0] + c * fp[0] - s * fp[1],
                         self.base_xy[1] + s * fp[0] + c * fp[1], fp[2]])

    def arm_to(self, world_pos, rot=None):
        """Set right-arm joints so the gripper reaches a world position with the grasp
        orientation, via MMK2Kdl IK (footprint frame). IK failures leave the arm held."""
        if rot is None:
            rot = self.grasp_rot
        fp = self.world_to_footprint(world_pos)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = fp
        ref = np.zeros(7)
        ref[0] = float(self.tc[2])
        ref[1:] = self.rarm_meas
        sols = self.kdl.inverse_kinematics(T_left=None, T_right=T, ref_pos=ref, target_height=float(self.slide_meas))
        if sols:
            self.tc[12:18] = np.asarray(sols[0])[1:7]
            self.arm_target_set = True
            return True
        else:
            self.get_logger().warn(f"IK unreachable: world={np.round(world_pos, 3)} fp={np.round(fp, 3)} (arm holds)")
            return False

    def ee_footprint_pose(self):
        """Actual gripper endpoint pose in footprint via measured joints."""
        _, T = self.kdl.forward_kinematics(np.concatenate([[float(self.slide_meas)], self.rarm_meas]), index="right")
        return T

    def ee_world(self):
        """Actual gripper endpoint in world via MMK2Kdl forward kinematics (measured joints)."""
        T = self.ee_footprint_pose()
        return self.footprint_to_world(T[:3, 3])

    # ---- smoothing ----
    def smooth_step(self):
        """Slew `action[2:19]` toward `tc[2:19]` so a freshly-set joint target ramps in
        instead of snapping (the cause of grasp 鐬Щ). When the target changes,
        normalize per-joint speed by the largest delta so all joints
        arrive together; then step_func each toward its target every tick."""
        if not np.allclose(self.tc[2:19], self.tc_prev[2:19]):
            dif = np.abs(self.action[2:19] - self.tc[2:19])
            self.joint_move_ratio[2:19] = dif / (np.max(dif) + 1e-6)
            self.joint_move_ratio[2] *= 0.3   # 鍗囬檷鏀炬參鍒?1/3: 鏀剧疆鏃剁墿浣撹交鏀句笅, 涓嶇牳妗岄潰寮曞彂鏅冨姩
            self.tc_prev[:] = self.tc
        step = self.joint_slew * self.dt
        for i in range(2, 19):
            self.action[i] = step_func(self.action[i], self.tc[i], self.joint_move_ratio[i] * step)

    # ---- publishing ----
    def publish(self):
        tw = Twist()
        tw.linear.x = float(self.tc[0])
        tw.linear.y = float(self.cur_lat)
        tw.angular.z = float(self.tc[1])
        self.cmd_vel_pub.publish(tw)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(data=[float(self.action[3]), float(self.action[4])]))
        self.larm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[5:11]] + [float(self.action[11])]))
        self.rarm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[12:18]] + [float(self.action[18])]))

    # ---- navigation ----
    def set_twist(self, lin, ang, lat=0.0):
        if getattr(self, "base_tilt", 0.0) > 0.6:
            # 底盘已倾倒: 切断所有速度指令, 等调度器/人工处理, 别再滚来滚去
            lin = ang = lat = 0.0
        self._st_dbg = getattr(self, "_st_dbg", 0) + 1
        if self._st_dbg <= 6:
            self.get_logger().info(f"SET_TWIST raw=({lin:.3f},{ang:.3f}) des=({float(np.clip(lin,-self.max_lin,self.max_lin)):.3f},{float(np.clip(ang,-self.max_ang,self.max_ang)):.3f})")
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))
        self.des_lat = float(np.clip(lat, -0.20, 0.20))

    def ramp_twist(self):
        """Acceleration-limit the published velocity so /cmd_vel changes smoothly."""
        dl = np.clip(self.des_lin - self.cur_lin, -self.max_lin_acc * self.dt, self.max_lin_acc * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -self.max_ang_acc * self.dt, self.max_ang_acc * self.dt)
        dt_lat = np.clip(self.des_lat - self.cur_lat, -self.max_lin_acc * self.dt, self.max_lin_acc * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.cur_lat += dt_lat
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang
        if getattr(self, "_rt_dbg", 0) < 5:
            self._rt_dbg = getattr(self, "_rt_dbg", 0) + 1
            self.get_logger().info(f"RAMPTWIST des=({self.des_lin:.3f},{self.des_ang:.3f}) cur=({self.cur_lin:.4f},{self.cur_ang:.4f}) dt={self.dt}")

    def follow_route(self, route, final_yaw):
        # ---- stuck-recovery: if base hasn't moved for ~3 s, reverse then rotate ----
        if self._stuck_recovering:
            elapsed = self.now() - self._stuck_recover_t0
            if self._stuck_recover_phase == "reverse":
                if elapsed < 1.0:
                    self.set_twist(-0.20, 0.0)
                    return False
                else:
                    self._stuck_recover_phase = "rotate"
                    self._stuck_recover_t0 = self.now()
            if self._stuck_recover_phase == "rotate":
                if elapsed < 1.5:
                    self.set_twist(0.0, 0.6)
                    return False
                else:
                    self._stuck_recovering = False
                    self._stuck_ticks = 0
                    self._stuck_prev_xy = None
                    self.nav_mode = "turn"
                    self.get_logger().info(
                        f"[stuck-recovery] resume nav after reverse+rotate "
                        f"(#{self._stuck_count})")

        if self.nav_idx < len(route):
            target = np.array(route[self.nav_idx], dtype=float)
            delta = target - self.base_xy
            dist = float(np.linalg.norm(delta))
            yaw_err = wrap_to_pi(math.atan2(delta[1], delta[0]) - self.base_yaw)

            # ---- detect stuck ----
            if self._stuck_prev_xy is not None and not self._stuck_recovering:
                disp = float(np.linalg.norm(self.base_xy - self._stuck_prev_xy))
                if disp < 0.03:
                    self._stuck_ticks += 1
                else:
                    self._stuck_ticks = 0
            self._stuck_prev_xy = self.base_xy.copy()

            if self._stuck_ticks >= 150 and not self._stuck_recovering:
                self._stuck_count += 1
                self.get_logger().warn(
                    f"[stuck-recovery] base stuck for {self._stuck_ticks} ticks "
                    f"(#{self._stuck_count}); reversing then rotating")
                self._stuck_recovering = True
                self._stuck_recover_t0 = self.now()
                self._stuck_recover_phase = "reverse"
                return False

            if self.nav_mode == "turn":
                self.set_twist(0.0, 2.2 * yaw_err)
                if abs(yaw_err) < self.turn_tol:
                    self.nav_mode = "drive"
            else:
                if dist < self.pos_tol:
                    self.nav_idx += 1
                    self.nav_mode = "turn"
                    self.set_twist(0.0, 0.0)
                    self._stuck_ticks = 0
                    self._stuck_prev_xy = None
                else:
                    if abs(yaw_err) < 0.05 or dist < 0.25:
                        ang = 0.0
                    else:
                        ang = 2.2 * yaw_err
                    align = max(0.0, math.cos(yaw_err))
                    self.set_twist(1.0 * dist * align, ang)
            return False
        yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
        self.set_twist(0.0, 1.8 * yaw_err)
        if abs(yaw_err) < self.turn_tol:
            self.set_twist(0.0, 0.0)
            return True
        return False

    def reset_nav(self):
        self.nav_idx = 0
        self.nav_mode = "turn"
        self._stuck_ticks = 0
        self._stuck_prev_xy = None
        self._stuck_recovering = False

    # ---- manipulation step gating (joint-space convergence + dwell) ----
    def action_done(self, dwell=0.4):
        if self.now() - self.state_t0 < dwell:
            return False
        slide_ok = abs(self.slide_meas - self.tc[2]) < 0.02
        arm_ok = (not self.arm_target_set) or np.max(np.abs(self.rarm_meas - self.tc[12:18])) < 0.05
        return slide_ok and arm_ok

    def deploy_done(self, dwell=0.6):
        if self.now() - self.state_t0 < dwell:
            return False
        slide_err = abs(self.slide_meas - self.tc[2])
        joint_err = np.max(np.abs(self.rarm_meas - self.tc[12:18])) if self.arm_target_set else float("inf")
        cart_err = float("inf")
        rot_err = float("inf")
        if self.DEPLOY_WORLD is not None:
            T = self.ee_footprint_pose()
            cart_err = float(np.linalg.norm(self.footprint_to_world(T[:3, 3]) - self.DEPLOY_WORLD))
            rot_delta = T[:3, :3].T @ self.grasp_rot
            cos_angle = np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
            rot_err = float(math.acos(cos_angle))

        # 收紧: 必须等手臂真正收敛(实测到目标 <~1.5cm)才进入 CREEP, 否则冻结的
        # "高位"姿态还在运动中, 会把接近高度/横向带偏(货架层板干涉的根源)。
        ready = slide_err < 0.03 and (
            joint_err < 0.05
            or (cart_err < 0.030 and rot_err < 0.15)
        )
        if not ready and self.now() - self.last_deploy_wait_log > 1.0:
            self.get_logger().info(
                f"[deploy] waiting: slide_err={slide_err:.3f} "
                f"joint_err={joint_err:.3f} cart_err={cart_err:.3f} rot_err={rot_err:.3f}")
            self.last_deploy_wait_log = self.now()
        return ready

    def enter_sub(self):
        self.sub_entered = True
        self.state_t0 = self.now()
        self.arm_target_set = False

    def run_sub(self, setter, n_states):
        if not self.sub_entered:
            setter(self.sub_idx)
            self.enter_sub()
        if self.action_done():
            self.sub_idx += 1
            self.sub_entered = False
        return self.sub_idx >= n_states

    # ---- main 30 Hz tick ----
    def tick(self):
        self._tcount += 1
        # 导航放弃标志(BIG 逃逸 3 次仍卡死): 退出 rc=1, 让调度器重置场景
        if getattr(self, "nav_abort", False) and not getattr(self, "_abort_done", False):
            self._abort_done = True
            self.get_logger().error(
                f"[abort] navigation aborted at base={self.base_xy}; exiting rc=1")
            import os as _os
            _os._exit(1)
        if self.base_xy is None or self.jpos is None:
            if not getattr(self, "_wait_dbg", False):
                self._wait_dbg = True
                self.get_logger().info(f"WAIT base_xy={self.base_xy} jpos_set={self.jpos is not None}")
            return
        # 预置已知箱阵到 DWA 地图(免去 lidar 逐帧建图导致的"先撞后绕")
        if not getattr(self, "_obs_seeded", False) and getattr(self, "planner", None) is not None:
            if self.planner.map_origin is None:
                # 启动即固定地图原点(14m 见方覆盖全场), 否则要等 DWA 进入
                # 障碍区才建图, 注入太晚(机器人已钻进窄缝才绕)。
                self.planner.map_origin = (self.base_xy[0] - self.planner.map_half,
                                           self.base_xy[1] - self.planner.map_half)
            self._obs_seeded = True
            try:
                import obstacle_layout as _ol
                _seed = int(os.getenv("OBSTACLE_SEED", "62"))
                _g = _ol.generate_obstacle_layout(seed=_seed)
                _boxes = [(p[0] - 0.96, p[1] - 1.01, _g.yaws[k], 0.6, 0.4)
                          for k, p in _g.positions.items()]
                # 墙注入(极小膨胀 0.05, seed_known_obstacles 里按 sx==0.1 识别):
                # 只为让动态走廊 A* 不穿墙(否则路径绕到墙外被边界检查拒绝,
                # 换种子就回退到 seed62 固定路径, 会撞新种子的箱子)。
                # 墙从 y=-5 到 y=3 封死南北两端, A* 无法绕墙 -> 路径必然在走廊内
                _boxes += [(0.53, -1.0, 0.0, 0.1, 8.0),   # 东墙 x=0.53
                           (-2.47, -1.0, 0.0, 0.1, 8.0)]  # 西墙 x=-2.47
                if self.seed_known_obstacles(_boxes):
                    self.get_logger().info(
                        f"[dwa] seeded {len(_boxes)} known obstacles (seed={_seed})")
            except Exception as _e:
                self.get_logger().warn(f"[dwa] obstacle seed failed: {_e}")
        if self._tcount % 150 == 0:
            self.get_logger().info(f"HEARTBEAT tick={self._tcount} phase={self.phase} base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) yaw={self.base_yaw:.2f} cmd=({self.tc[0]:.3f},{self.tc[1]:.3f}) navidx={self.nav_idx} navmode={self.nav_mode}")

        if self.phase == NAV_SHELF:
            # 从桌边(上一单的放置点)出发: 前置固定通道折线, 纯跟踪确定性
            # 通过箱阵回到货架车道(避开 A*/DWA 在桌角的失效)。
            if not getattr(self, "_corr_prefix_done", False):
                self._corr_prefix_done = True
                if self.base_xy[1] < -2.0 and self.base_xy[0] < 0.5:
                    _route = self._corridor_route(True)
                    if _route is None:
                        self.get_logger().warn(
                            "[nav] corridor route unavailable; using fixed fallback")
                        _route = FALLBACK_CORRIDOR
                    prefix = [list(_route[-1])] + _route[::-1]  # 桌边(南) -> 车道(北)
                    self.route_to_shelf = prefix + list(self.route_to_shelf)
                    self._scripted_until = len(prefix)
                    self.scripted_path = True
                    self.get_logger().info(
                        "[nav] table start: prepend fixed corridor path "
                        f"({len(prefix)} waypoints)")
            if (getattr(self, "scripted_path", False)
                    and self.nav_idx >= getattr(self, "_scripted_until", 0)):
                self.scripted_path = False
            if self.follow_route(self.route_to_shelf, GRASP_YAW):
                self.phase, self.deploy_set = DEPLOY, False
                self.det_buf.clear()
                self.target_locked = False
                self.OBJECT_WORLD = None
                self.DEPLOY_WORLD = None
                self.CREEP_STOP_Y = None
                self.state_t0 = self.now()   # start the detection dwell window fresh
                self._search_step = 0
                self._search_t0 = self.now()
                self._plan_idx = 0
                self._plan_loops = 0
                self._search_failed = False
                self.search_level = self.prior["level"]
                if self.skip_grasp:
                    self.phase = NAV_TABLE
                    self.get_logger().info("grasp skipped; going straight to delivery table")
                    self.reset_nav()
        elif self.phase == DEPLOY:
            # 鍦ㄩ粍绾块檮杩戞妸鑳宠唺鎽嗘垚鎶撳彇濮挎€?涔嬪悗闈犲簳鐩樼洿绾?creep 瀹屾垚杩涚粰
            self.set_twist(0.0, 0.0)
            if not self.deploy_set:
                # Aim head/slide so the shelf is in view, accumulate
                # detections, then pose the arm from the vision target.
                self.tc[4] = self.head_pitch
                self.tc[2] = self.slide_grasp
                self.tc[18] = GRIP_OPEN
                if self.target_locked:
                    # Locked: settle the spine, then pose the arm at the
                    # deploy pose (商品南 ~0.22m, 货架外). The EE-driven CREEP
                    # afterwards brings the base in to wrap the product.
                    if getattr(self, "_deploy_t0", None) is None:
                        self._deploy_t0 = self.now()
                    if abs(self.slide_meas - self.tc[2]) > 0.15:
                        if self.now() - self._deploy_t0 > 15.0:
                            self.get_logger().warn(
                                "[deploy] slide settle timeout; forcing deploy")
                            self.deploy_set = True
                            self.state_t0 = self.now()
                    elif self._align_to_reach():
                        pass   # IK 不可达: 横移/前探让 DEPLOY 点落入可达窗口
                    elif self.arm_to(self.DEPLOY_WORLD):
                        self.deploy_set = True
                        self._deploy_t0 = None
                        self.state_t0 = self.now()
                    else:
                        self.target_locked = False
                        self.det_buf.clear()
                elif self.gt_target:
                    # GT 模式: 直接使用布局真值锁定目标(第一阶段验证机械臂链路用)
                    self.OBJECT_WORLD = self.slot_world.copy()
                    self._send_grasp_target()
                    self.DEPLOY_WORLD = self.OBJECT_WORLD + self.deploy_offset
                    self.CREEP_STOP_Y = float(self.OBJECT_WORLD[1]) + self.creep_dy
                    self.target_locked = True
                    self.get_logger().info(
                        f"[GT] lock OBJECT={np.round(self.OBJECT_WORLD,3)} "
                        f"DEPLOY={np.round(self.DEPLOY_WORLD,3)} "
                        f"CREEP_STOP_Y={self.CREEP_STOP_Y:.3f}")
                elif self.now() - self.state_t0 < DETECT_DWELL:
                    pass
                elif self.prior_fallback and (SEARCH_SKIP or getattr(self, "_search_failed", False)):
                    # 视觉域不匹配(开发期): 关闭搜索或搜索耗尽后用布局先验位置兜底,
                    # 让取送流程继续跑完(录像/联调用), 抓取可能抓空。
                    self._lock_from_prior()
                elif self._view_search():
                    pass   # searching: back off to see the shelf + align + sweep levels
                elif getattr(self, "_search_failed", False) and self.prior_fallback:
                    self._lock_from_prior()
                else:
                    self._lock_target()
            if self.deploy_set and self.deploy_done():
                self.phase = CREEP
        elif self.phase == CREEP:
            # 两段式: (1) 保持高位(夹爪在瓶顶之上), 底盘边走边横向对正, 直到夹爪
            # 正对物体正上方; (2) 冻结底盘, 张爪竖直慢降, 并低速闭环对中, 让瓶身
            # 从两指间进入后再闭夹(侧夹姿态不能垂直砸瓶顶)。
            if getattr(self, "_creep_t0", None) is None:
                self._creep_t0 = self.now()
                self._desc_t0 = None
            obj = self.OBJECT_WORLD
            ee = self.ee_world()
            dx = float(obj[0]) + float(self.deploy_offset[0]) - ee[0]
            vy = float(np.clip(-dx * 0.8, -0.15, 0.15))
            if abs(dx) < 0.015:
                vy = 0.0
            # 官方几何: EE(指尖)停在物体中心以北 CREEP_STOP_DY, 手指向后包住瓶身
            # (停太浅只有指尖蹭到瓶子前半, 闭夹抓不住还会带倒瓶子)。
            above_y = float(obj[1]) + float(self.creep_dy)
            timeout = (self.now() - self._creep_t0) > 18.0
            if ee[1] < above_y and not timeout:
                self.set_twist(CREEP_SPEED,
                               CREEP_YAW_KP * wrap_to_pi(GRASP_YAW - self.base_yaw), vy)
                if self.now() - getattr(self, "_last_clog", 0) > 1.0:
                    self.get_logger().info(
                        f"[creep] above ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) dx={dx:+.3f} vy={vy:+.3f}")
                    self._last_clog = self.now()
            else:
                self.set_twist(0.0, 0.0)
                if self._desc_t0 is None:
                    self._desc_t0 = self.now()
                desired = obj + np.array([float(self.deploy_offset[0]),
                                         float(self.creep_dy), 0.0])
                ee = self.ee_world()
                err = desired - ee
                corr = np.clip(err, -0.04, 0.04)
                self.arm_to(desired + 0.5 * corr)     # 过量下发抵消伺服残差
                ee = self.ee_world()
                fine_err = float(np.linalg.norm((desired - ee)[:2]))
                err_z = abs(float(ee[2]) - float(desired[2]))
                settle = (self.now() - self._desc_t0) > 3.5
                if self.now() - getattr(self, "_last_dlog", 0) > 0.5:
                    self.get_logger().info(
                        f"[descend] ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) err_xy={fine_err:.3f} err_z={err_z:.3f} settle={settle}")
                    self._last_dlog = self.now()
                # Z 门限: EE 必须先降到物体中心高度附近再闭夹, 否则手指在瓶口
                # 上方闭合会蹭倒瓶子(gt_align1 里 z=1.003 vs obj 0.923)。
                if (fine_err < 0.015 and err_z < 0.03) or settle or timeout:
                    self.set_twist(0.0, 0.0)
                    self.phase = CLOSE
                    self._creep_t0 = None
                    self._desc_t0 = None
                    self.state_t0 = self.now()
        elif self.phase == CLOSE:
            self.set_twist(0.0, 0.0)
            self.tc[18] = self.grip_close
            if self.now() - self.state_t0 > float(os.getenv("CLOSE_WAIT", "0.8")):
                self.phase = LIFT
        elif self.phase == LIFT:
            # 绔栫洿鎶€璧?鍑忓皬 slide,鑳稿儴涓婄Щ),璁╃墿浣撶鍏ラ殧鏉?鑳宠唺鍏妭淇濇寔涓嶅姩
            self.set_twist(0.0, 0.0)
            if getattr(self, "_lift_t0", None) is None:
                self._lift_t0 = self.now()
            self.tc[2] = self.slide_grasp - LIFT_AMOUNT
            # slide 贴近货架时有系统性偏差(接触/载荷/上限), L1 档实测可达 0.13+,
            # 门限放宽到 0.16 并加 6s 超时, 否则流程会永久卡在 LIFT 被调度器杀掉。
            if (abs(self.slide_meas - self.tc[2]) < 0.16
                    or (self.now() - self._lift_t0) > 6.0):
                self._lift_t0 = None
                self.phase = RETREAT
        elif self.phase == RETREAT:
            yaw_err = wrap_to_pi(GRASP_YAW - self.base_yaw)
            # 鍊掕溅(淇濇寔鎶撳彇鏈濆悜)閫€鍥為粍绾夸腑鐐?object 杩樺す鍦ㄦ墜閲?            yaw_err = wrap_to_pi(GRASP_YAW - self.base_yaw)
            if self.base_xy[1] > YELLOW_MID_Y + self.pos_tol:
                self.set_twist(-RETREAT_SPEED, 1.0 * yaw_err)
            else:
                self.set_twist(0.0, 0.0)
                self.state_t0 = self.now()
                if self.finish_after_retreat:
                    self.get_logger().info(
                        "retreat done; simplified baseline complete with grasp retained")
                    self.phase = DONE
                else:
                    held = self._check_held()
                    self.held_ok = held
                    if not held and self.grasp_attempts < GRASP_MAX_ATTEMPTS:
                        # failed to hold: go back and retry the deploy/grasp once
                        self.grasp_attempts += 1
                        self.get_logger().warn(
                            f"grasp not held — retry #{self.grasp_attempts}")
                        self.phase = DEPLOY
                        self.deploy_set = False
                        self.target_locked = False
                        self.det_buf.clear()
                        self.state_t0 = self.now()
                        # 重试要重置搜索状态, 否则上一轮"计划耗尽"的标记会
                        # 让搜索直接短路(_search_failed 残留 → 不移动只空等)。
                        self._plan_idx = 0
                        self._plan_loops = 0
                        self._search_failed = False
                        self._search_t0 = self.now()
                        self.search_level = self.prior["level"]
                    else:
                        # 保持夹取姿势运输(即使抓空也要模拟夹取送货的观感);
                        # 姿态补偿: 滑轨整体下移 10cm 降重心(手臂关节姿态不变,
                        # 观感仍是夹取), 配合稳定代价改善绕障时的重心偏离。
                        self.tc[2] = max(0.0, float(self.slide_grasp) - 0.10)
                        self.get_logger().info(
                            f"retreat done; object_held={held}; keep grasp pose, "
                            f"lower slide to {self.tc[2]:.2f} for transport stability")
                        self.phase = NAV_TABLE
                        self.reset_nav()
        elif self.phase == NAV_TABLE:
            # 返程固定通道路径: 去程验证过的折线反向复用, 绕开箱阵窄缝
            # (返程卡死都发生在 box_04/box_03 之间的膨胀区), 终点仍是
            # 原来的桌子定点(放置精度优先)。
            if not getattr(self, "_table_path_set", False):
                self._table_path_set = True
                # 首点=机器人当前位置: 否则 DWA 从 0.4m 外接近 (-1.20,2.30)
                # 会提前判定"到达"停住 -> 车道出口卡死翻车。
                _route = self._corridor_route(False)
                if _route is None:
                    self.get_logger().warn(
                        "[nav] corridor route unavailable; using fixed fallback")
                    _route = FALLBACK_CORRIDOR
                self._table_route = (
                    [[float(self.base_xy[0]), float(self.base_xy[1])]]
                    + [list(p) for p in _route]          # 车道(北) -> 桌边(南)
                    + [[-1.94, -3.10]]
                )
                self._table_scripted_until = len(self._table_route)
                self.scripted_path = True
                self.nav_idx = 0
                self.get_logger().info(
                    "[nav] table return: fixed corridor path (reverse of outbound)")
            if (getattr(self, "scripted_path", False)
                    and self.nav_idx >= getattr(self, "_table_scripted_until", 0)):
                self.scripted_path = False
            if self.follow_route(getattr(self, "_table_route", ROUTE_TO_TABLE), YAW_SOUTH):
                # Switch to place rotation BEFORE entering PLACE phase
                # so the arm has time to move while the slide adjusts
                self.grasp_rot_save = self.grasp_rot.copy()
                self.grasp_rot = PLACE_ROTATION
                self.arm_to(TABLE_ORIGIN + np.array([0.0, 0.0, 0.05]))
                self.phase, self.place_sub = PLACE, 0
                self.state_t0 = self.now()
        elif self.phase == PLACE:
            self.set_twist(0.0, 0.0)
            if self.place_sub == 0:
                # 鍏堟妸鍗囬檷骞冲彴鏁翠綋闄嶄笅鏉?鐗╀綋闅忎箣绔栫洿涓嬮檷鍒版闈㈤檮杩?鎵嬭噦鍏宠妭涓嶅姩)
                self.tc[2] = PLACE_LOWER_SLIDE
                if abs(self.slide_meas - PLACE_LOWER_SLIDE) < 0.02:
                    self.place_sub = 1
                    self.state_t0 = self.now()
            elif self.place_sub == 1:
                # 鍒颁綅鍚庢澗鐖?鐗╀綋钀芥
                self.tc[18] = GRIP_OPEN
                if self.now() - self.state_t0 > 1.0:
                    self.place_sub = 2
                    self.state_t0 = self.now()
            else:
                # 松开后倒车离桌(朝南时负速度=往北退): 下一单从桌边死角
                # 起步时 DWA/A* 找不到路 -> 卡死+连续倒车翻车。
                if self.base_xy[1] < -2.35 and self.now() - self.state_t0 < 6.0:
                    self.set_twist(-0.15, 0.0)
                else:
                    self.phase = DONE
        else:
            self.set_twist(0.0, 0.0)
            if not getattr(self, "reported", False):
                self.reported = True
                ee = self.ee_world()
                obj = self.OBJECT_WORLD
                self.get_logger().info(
                    f"[OUTCOME] pipeline=DONE locked={self.target_locked} "
                    f"slot={np.round(self.slot['world_position'],3)} "
                    f"OBJECT={np.round(obj,3) if obj is not None else None} "
                    f"gripper_final=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                    f"grip_close={self.grip_close} object_held={self.held_ok} lift_delta={LIFT_AMOUNT}")
                self.state_t0 = self.now()
            elif self.now() - self.state_t0 > 1.0:
                self.get_logger().info("[OUTCOME] finished; shutting down")
                # hard-exit so the task scheduler sees the process end and moves
                # on to the next order (rclpy.shutdown alone left threads alive).
                import os as _os
                _os._exit(0)

        self.ramp_twist()
        self.smooth_step()
        self.publish()

        try:
            if self.now() - self.last_log > 1.0:
                ee = self.ee_world()
                obj = self.OBJECT_WORLD
                obj_str = (f"({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})"
                           if obj is not None else "unlocked")
                self.get_logger().info(
                    f"phase={PHASE_NAME[self.phase]} sub={self.sub_idx} "
                    f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) yaw={self.base_yaw:.2f} slide={self.slide_meas:.3f} "
                    f"gripper=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                    f"obj={obj_str} locked={self.target_locked} kind={self.kind} "
                    f"cmd_arm={np.round(self.tc[12:18], 3)} meas_arm={np.round(self.rarm_meas, 3)}"
                    + (f" GS_z={self.grasp_status[5]:.3f} GS_tilt={self.grasp_status[7]:.1f} "
                       f"GS_grip={int(self.grasp_status[2])}"
                       if self.grasp_status is not None else " GS=none"))
                self.last_log = self.now()
        except Exception as e:
            self.get_logger().error(f"TICK EXCEPTION: {type(e).__name__}: {e}")

    def _send_grasp_target(self):
        """告诉服务端目标商品编号(product_NNN -> 运行时 item_run_*_NNN)。"""
        body = self.slot.get("body", "")
        nn = 0
        if body.startswith("product_"):
            nn = int(body.split("_")[1])
        self.grasp_target_pub.publish(Float64(data=float(nn)))
        self.get_logger().info(f"[grasp] sent grasp_target NN={nn} (body={body})")

    def _align_to_reach(self):
        """IK 不可达时的对齐恢复: 横移/前探让 DEPLOY 点落入可达窗口。

        可达窗口(底盘系, R_ARM_SIDE 侧抓):
          fwd ∈ [0.35, 0.60]   (前向: 手臂够得到)
          lat ∈ [-0.20, +0.30] (横向: 手臂侧向范围)
        返回 True 表示还在对齐移动, False 表示已进入窗口(可重试 arm_to)。
        """
        if self.DEPLOY_WORLD is None or self.base_xy is None:
            return False
        fp = self.world_to_footprint(self.DEPLOY_WORLD)
        fwd, lat = fp[0], fp[1]
        # DEPLOY 阶段手臂在商品前 ~0.22m 的自由空间, 这里先把横向对正(lat->0),
        # 之后 CREEP 只走直线, 就不会在商品深度处横移扫倒瓶子。
        fwd_ok = 0.30 <= fwd <= 0.62
        lat_ok = abs(lat) < 0.03
        if fwd_ok and lat_ok:
            return False
        # 横向纠偏: 目标在东(lat>0)则底盘往东(vy<0, 底盘+y=西)
        vy = float(np.clip(lat * 0.8, -0.15, 0.15))
        if abs(lat) < 0.03:
            vy = 0.0
        # 前向纠偏: 目标太近(fwd<0.35)则后退, 太远(fwd>0.60)则前进
        lin = 0.0
        if fwd < 0.35:
            lin = -0.06
        elif fwd > 0.60:
            lin = 0.06
        self.set_twist(lin, CREEP_YAW_KP * wrap_to_pi(GRASP_YAW - self.base_yaw), vy)
        self.get_logger().info(
            f"[reach] align: fwd={fwd:.2f} lat={lat:.2f} lin={lin:+.2f} vy={vy:+.2f}")
        return True

    def _check_held(self):
        """Grasp success: 夹爪闭合只是"触发条件", 真正判定用服务端真值:
        两指接触 + 物体被提升 + 相对稳定 + 连续满足若干 tick。

        grasp_status = [left_touch, right_touch, gripped, x, y, z, speed, tilt_deg]
        """
        # 触发条件: 夹爪闭合被物体撑住(手指合不拢)
        g = self.jpos.get("right_arm_eef_gripper_joint", None)
        trigger = (g is not None and float(g) > float(self.grip_close) + 0.10)
        st = self.grasp_status
        if st is None or len(st) < 8:
            self.get_logger().info("[held] no grasp_status yet (waiting for contact topic)")
            return False
        left, right, gripped = bool(st[0]), bool(st[1]), bool(st[2])
        z, speed, tilt = st[5], st[6], st[7]
        # 提升: 物体高度高于货架层(已离开货位) — 用初始 z 记录判断
        lifted = z > 0.42 and z < 1.40
        stable = speed < 0.25 and tilt < 30.0
        ok = trigger and gripped and lifted and stable
        self.grasp_ok_ticks = self.grasp_ok_ticks + 1 if ok else 0
        self.get_logger().info(
            f"[held] trigger={trigger} left={left} right={right} gripped={gripped} "
            f"z={z:.3f} speed={speed:.3f} tilt={tilt:.1f} ok_ticks={self.grasp_ok_ticks}")
        return self.grasp_ok_ticks >= 5   # 30Hz 下连续 ~0.17s 稳定


def main():
    parser = argparse.ArgumentParser(description="supermarket sorting pick-place client")
    parser.add_argument("--kind", default="kele", help="product kind to pick (kele/maidong/...)")
    parser.add_argument("--layout", default=None, help="path to retail_competition_layout.json")
    parser.add_argument("--finish-after-retreat", action="store_true")
    parser.add_argument("--skip-grasp", action="store_true")
    parser.add_argument("--gt-target", action="store_true",
                        help="第一阶段: 直接用布局真值作为目标位置(不依赖视觉)")
    parser.add_argument("--shelf", default=None, help="限定货架 (A/B/C/D/E), 默认第一个")
    parser.add_argument("--prior-fallback", action="store_true",
                        help="视觉搜索失败后用布局先验位置兜底(开发/录像用)")
    args = parser.parse_args()

    rclpy.init()
    node = PickPlaceClient(kind=args.kind, shelf=args.shelf, layout_path=args.layout,
                           finish_after_retreat=args.finish_after_retreat,
                           skip_grasp=args.skip_grasp, gt_target=args.gt_target,
                           prior_fallback=args.prior_fallback)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
