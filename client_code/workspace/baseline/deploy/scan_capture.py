#!/usr/bin/env python3
"""Scan the shelf with the head camera and save per-kind product images.

Runs in the client container on ROS_DOMAIN_ID=96.  Drives the base to the shelf
lane, strafes west along it while sweeping the head pitch (and slowly the
spine), and saves:
  <out>/<kind>/NNN.jpg   tight crops around each GT detection of that kind
  <out>/frames/NNNN.jpg  full frames (for the video / YOLO auto-labelling)
  <out>/scan.mp4         the full scan video
  <out>/meta.json        kind / aruco_id / world position for every crop

Uses the GT perception node's detections (pixel bbox in the Detection3D) to
locate each product; no ground-truth file is read here.
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from vision_msgs.msg import Detection3DArray

LANE_Y = 2.30
SHELF_YAW = math.pi / 2.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ScanCapture(Node):
    def __init__(self, out_dir):
        super().__init__("scan_capture")
        self.out = out_dir
        self.bridge = CvBridge()
        self.rgb = None
        self.dets = []
        self.x = self.y = self.yaw = None
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.head_pub = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.spine_pub = self.create_publisher(
            Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.create_subscription(Image, "/head_camera/color/image_raw",
                                 self.rgb_cb, 5)
        self.create_subscription(Detection3DArray, "/kele/detections",
                                 self.det_cb, 5)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 5)
        os.makedirs(self.out, exist_ok=True)

    def rgb_cb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def det_cb(self, msg):
        self.dets = msg.detections

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def cmd(self, lin, lat, ang):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(lin), float(lat), float(ang)
        self.cmd_pub.publish(t)

    def head(self, pitch, yaw=0.0):
        m = Float64MultiArray()
        m.data = [float(yaw), float(pitch)]
        self.head_pub.publish(m)

    def slide(self, s):
        m = Float64MultiArray()
        m.data = [float(s)]
        self.spine_pub.publish(m)

    def spin(self, sec):
        t0 = time.time()
        while time.time() - t0 < sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def drive_to(self, tx, ty, tol=0.12, timeout=70.0):
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.x is None:
                continue
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            err = wrap(math.atan2(dy, dx) - self.yaw)
            lin = min(0.40, 1.4 * dist) * max(0.25, math.cos(err))
            self.cmd(lin, 0.0, 1.8 * err)
        self.cmd(0.0, 0.0, 0.0)
        self.spin(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/baseline/product_images")
    ap.add_argument("--speed", type=float, default=0.07, help="strafe speed m/s")
    ap.add_argument("--x-start", type=float, default=1.90)
    ap.add_argument("--x-end", type=float, default=-1.85)
    ap.add_argument("--crop-m", type=float, default=0.20, help="crop box size in metres")
    args = ap.parse_args()

    rclpy.init()
    node = ScanCapture(args.out)
    fx = 600.0
    for kind in ("sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
                 "kele", "kouxiangtang", "pingguo", "chengzi"):
        os.makedirs(os.path.join(args.out, kind), exist_ok=True)
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    print("[scan] waiting for odom/rgb ...", flush=True)
    t0 = time.time()
    while (node.x is None or node.rgb is None) and time.time() - t0 < 30:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"[scan] start ({node.x:.2f},{node.y:.2f}) yaw={node.yaw:.2f}; driving to lane", flush=True)
    node.slide(0.20)
    node.head(-0.60)
    node.drive_to(1.92, 2.05, timeout=90)
    node.drive_to(1.90, LANE_Y, timeout=60)
    node.drive_to(1.60, LANE_Y, timeout=40)

    print("[scan] strafe west along the shelf", flush=True)
    counts = {}
    meta = []
    frames = []
    fi = 0
    t0 = time.time()
    video = None
    while rclpy.ok() and node.x is not None and node.x > args.x_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        el = time.time() - t0
        pitch = -0.65 + 0.40 * math.sin(2 * math.pi * el / 7.0)
        slide = 0.15 + 0.13 * math.sin(2 * math.pi * el / 23.0)
        node.head(pitch)
        node.slide(slide)
        yaw_err = wrap(SHELF_YAW - node.yaw)
        node.cmd(0.0, args.speed, 2.0 * yaw_err)
        if node.rgb is None:
            continue
        img = node.rgb.copy()
        if video is None:
            h, w = img.shape[:2]
            video = cv2.VideoWriter(os.path.join(args.out, "scan.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        video.write(img)
        if fi % 3 == 0:
            cv2.imwrite(os.path.join(args.out, "frames", f"{fi:05d}.jpg"), img)
        fi += 1
        for det in node.dets:
            if not det.results:
                continue
            kind = det.results[0].hypothesis.class_id
            aruco = next((r.hypothesis.class_id for r in det.results
                          if r.hypothesis.class_id.startswith("aruco:")), None)
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y
            if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
                continue
            # keep only well-centred projections: the pinhole projection is
            # accurate near the optical axis, while lens distortion shifts the
            # pixel near the frame edges (crops there miss the product).
            if not (0.18 * img.shape[1] < u < 0.82 * img.shape[1]
                    and 0.18 * img.shape[0] < v < 0.82 * img.shape[0]):
                continue
            side = int(max(56, min(200, args.crop_m * fx / 0.9)))
            x0, x1 = int(max(0, u - side / 2)), int(min(img.shape[1], u + side / 2))
            y0, y1 = int(max(0, v - side / 2)), int(min(img.shape[0], v + side / 2))
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            crop = img[y0:y1, x0:x1]
            n = counts.get(kind, 0)
            if n >= 40:
                continue
            cv2.imwrite(os.path.join(args.out, kind, f"{n:03d}.jpg"), crop)
            counts[kind] = n + 1
            p = det.results[0].pose.pose.position
            meta.append({"kind": kind, "aruco": aruco, "file": f"{kind}/{n:03d}.jpg",
                         "world": [p.x, p.y, p.z],
                         "base": [node.x, node.y, node.yaw]})
        if int(el) % 5 == 0 and int(el * 10) % 10 == 0:
            pass
    node.cmd(0.0, 0.0, 0.0)
    if video is not None:
        video.release()
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[scan] done. frames={fi} crops={counts}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
