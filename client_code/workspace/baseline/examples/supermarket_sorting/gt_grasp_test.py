#!/usr/bin/env python3
"""GT 真值 → 机械臂 pre-grasp 最小闭环测试（第一阶段，不做视觉）。

流程:
  1. 从 retail_competition_layout.json 读指定商品的 GT world_position
  2. 读取 odom 底盘位姿
  3. 简单 P 控制把底盘开到 (slot_x, 2.475)，面向 +Y
  4. 用现有 MMK2Kdl IK 把 world 目标转成机械臂关节角并发布
  5. 停在 pre-grasp（不闭夹），打印 target_world / target_arm / current_arm / error

用法:
  python3 gt_grasp_test.py --kind maidong --approach-y 2.475
"""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from grasp_config import get_grasp
from mmk2_kdl import MMK2Kdl

YELLOW_MID_Y = 2.475
YAW_NORTH = math.pi / 2.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GtGraspTest(Node):
    def __init__(self, kind, shelf, approach_y, safe=True):
        super().__init__("gt_grasp_test")
        here = os.path.dirname(os.path.abspath(__file__))
        layout = json.load(open(os.path.join(here, "retail_competition_layout.json")))
        slots = [s for s in layout if s.get("object_kind") == kind]
        if shelf:
            slots = [s for s in slots if s.get("shelf") == shelf]
        self.slot = slots[0] if slots else None
        if self.slot is None:
            raise RuntimeError(f"no slot for kind={kind} shelf={shelf}")
        self.kind = kind
        self.target_world = np.array(self.slot["world_position"], dtype=float)
        self.approach_y = approach_y
        self.safe = safe

        g = get_grasp(kind)
        self.grasp_rot = np.array(g["grasp_rot"], dtype=float)
        self.slide = float({"L1": 0.30, "L2": 0.15, "L3": 0.00}.get(
            str(self.slot.get("level", "L2")), 0.15))
        self.head_pitch = float({"L1": -1.00, "L2": -0.70, "L3": -0.40}.get(
            str(self.slot.get("level", "L2")), -0.70))

        self.kdl = MMK2Kdl()
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(
            Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(
            Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)

        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)

        self.get_logger().info(
            f"[GT TARGET] slot={self.slot['shelf']}{self.slot['level']}{self.slot['column']} "
            f"kind={kind} world_position=({self.target_world[0]:.3f},{self.target_world[1]:.3f},"
            f"{self.target_world[2]:.3f})")
        self.get_logger().info(
            f"[ARM CFG] slide={self.slide} head_pitch={self.head_pitch} "
            f"level={self.slot['level']}")

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                   1 - 2 * (q.y * q.y + q.z * q.z))

    def js_cb(self, msg):
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name)
                     if i < len(msg.position)}

    def cmd(self, lin, lat, ang):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(lin), float(lat), float(ang)
        self.cmd_pub.publish(t)

    def rarm_meas(self):
        return np.array([self.jpos.get(f"right_arm_joint{i+1}", 0.0) for i in range(6)])

    def world_to_footprint(self, p_world):
        d = np.array(p_world, float) - np.array([self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def footprint_to_world(self, fp):
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([self.base_xy[0] + c * fp[0] - s * fp[1],
                         self.base_xy[1] + s * fp[0] + c * fp[1], fp[2]])

    def arm_to(self, world_pos, rot=None):
        rot = rot if rot is not None else self.grasp_rot
        fp = self.world_to_footprint(world_pos)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = fp
        ref = np.zeros(7)
        ref[0] = self.slide
        ref[1:] = self.rarm_meas()
        sols = self.kdl.inverse_kinematics(T_left=None, T_right=T, ref_pos=ref,
                                           target_height=self.slide)
        if not sols:
            self.get_logger().warn(f"[ARM] IK unreachable: world={np.round(world_pos,3)} "
                                   f"fp={np.round(fp,3)}")
            return None
        joints = np.asarray(sols[0])[1:7]
        self.get_logger().info(f"[ARM TARGET] world=({world_pos[0]:.3f},{world_pos[1]:.3f},"
                               f"{world_pos[2]:.3f}) fp=({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f}) "
                               f"joints={np.round(joints,3)}")
        return joints

    def ee_world(self):
        _, T = self.kdl.forward_kinematics(
            np.concatenate([[self.slide], self.rarm_meas()]), index="right")
        return self.footprint_to_world(T[:3, 3])

    def drive_to(self, tx, ty, tyaw, tol=0.10, timeout=120.0):
        self.get_logger().info(f"[DRIVE] -> ({tx:.2f}, {ty:.2f}) yaw={tyaw:.2f}")
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.base_xy is None:
                continue
            dx, dy = tx - self.base_xy[0], ty - self.base_xy[1]
            dist = math.hypot(dx, dy)
            yaw_e = wrap(tyaw - self.base_yaw)
            if dist < tol and abs(yaw_e) < 0.06:
                break
            lin = min(0.30, 1.1 * dist) * max(0.25, math.cos(wrap(math.atan2(dy, dx) - self.base_yaw)))
            self.cmd(max(-0.25, min(0.25, lin)), 0.0, max(-0.8, min(0.8, 1.6 * yaw_e)))
        self.cmd(0.0, 0.0, 0.0)
        rclpy.spin_once(self, timeout_sec=0.3)
        self.get_logger().info(f"[DRIVE] arrived at ({self.base_xy[0]:.3f},{self.base_xy[1]:.3f})")

    def strafe_to(self, tx, ty, tol=0.06, timeout=90.0):
        """面向货架(+Y)横向移动(linear.y)到指定 x, 保持 y 不变。"""
        self.get_logger().info(f"[STRAFE] x -> {tx:.2f}")
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.base_xy is None:
                continue
            x_err = tx - self.base_xy[0]
            yaw_e = wrap(YAW_NORTH - self.base_yaw)
            if abs(x_err) < tol:
                break
            # base +y(左) = 世界 -x(西): 往西(x_err<0)要 vy>0
            vy = float(np.clip(-x_err * 0.8, -0.15, 0.15))
            self.cmd(0.0, vy, max(-0.6, min(0.6, 1.5 * yaw_e)))
        self.cmd(0.0, 0.0, 0.0)
        rclpy.spin_once(self, timeout_sec=0.3)
        self.get_logger().info(f"[STRAFE] x={self.base_xy[0]:.3f}")

    def run(self):
        t0 = time.time()
        while time.time() - t0 < 8 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.base_xy is None or self.jpos is None:
            self.get_logger().error("[ROBOT] no odom/joint_states")
            return
        self.get_logger().info(
            f"[ROBOT] base=({self.base_xy[0]:.3f},{self.base_xy[1]:.3f}) yaw={self.base_yaw:.3f}")

        # 1) 底盘开到目标槽正前方: 两段式(避开障碍走廊) — 先沿 x=1.92 北上到车道,
        #    再横移到槽位正上方, 最后北上到"商品前 0.60m"(手臂 0.6m 能伸到,
        #    轮子距货架面还有 ~0.3m, 不会卡进货架)
        tx = float(self.slot["world_position"][0])
        grasp_y = float(self.target_world[1]) - 0.60
        self.drive_to(1.92, 2.30, YAW_NORTH)
        self.strafe_to(tx, 2.30)
        self.drive_to(tx, grasp_y, YAW_NORTH)

        # 2) 安全校验
        fp = self.world_to_footprint(self.target_world)
        fwd, lat, z = fp[0], fp[1], self.target_world[2]
        self.get_logger().info(
            f"[SAFETY] target_fp=({fwd:.3f},{lat:.3f},{z:.3f}) "
            f"level_z_ok={0.42 <= z <= 1.35} reach_ok={0.30 <= fwd <= 1.50} "
            f"lat_ok={abs(lat) <= 0.40}")
        if self.safe and not (0.42 <= z <= 1.35 and 0.30 <= fwd <= 1.50
                              and abs(lat) <= 0.40):
            self.get_logger().error("[SAFETY] target out of reach — NOT commanding arm")
            return

        # 3) 设置 slide + 头部到目标层
        for _ in range(30):
            sp = Float64MultiArray(data=[self.slide])
            self.spine_pub.publish(sp)
            hp = Float64MultiArray(data=[0.0, self.head_pitch])
            self.head_pub.publish(hp)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"[ARM] slide->{self.slide} head_pitch->{self.head_pitch}")

        # 4) pre-grasp: 商品上方 0.08m 且南移 0.12m(指尖能伸到商品), 不闭夹
        pre = self.target_world + np.array([0.0, -0.12, 0.08])
        joints = self.arm_to(pre)
        if joints is None:
            return
        msg = Float64MultiArray(data=list(joints) + [0.99])   # 夹爪保持张开
        t1 = time.time()
        while time.time() - t1 < 6.0 and rclpy.ok():
            self.rarm_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        # 5) 报告
        ee = self.ee_world()
        err = np.linalg.norm(ee - pre)
        self.get_logger().info(
            f"[RESULT] target_world=({pre[0]:.3f},{pre[1]:.3f},{pre[2]:.3f}) "
            f"current_arm=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
            f"error={err*1000:.0f}mm")
        self.get_logger().info("[RESULT] pre-grasp reached; gripper stays OPEN (safe mode)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="maidong")
    ap.add_argument("--shelf", default="C")
    ap.add_argument("--approach-y", type=float, default=YELLOW_MID_Y)
    ap.add_argument("--no-safe", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = GtGraspTest(args.kind, args.shelf, args.approach_y, safe=not args.no_safe)
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(f"[FAIL] {type(e).__name__}: {e}")
    node.cmd(0.0, 0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()