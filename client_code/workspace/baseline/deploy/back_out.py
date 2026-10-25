#!/usr/bin/env python3
"""从当前停靠点直接倒车(保持朝北)退出到车道, 验证能正常离开货架。"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class BackOut(Node):
    def __init__(self, target_y=2.30):
        super().__init__("back_out")
        self.target_y = target_y
        self.x = self.y = self.yaw = None
        self.pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.od, 5)

    def od(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def cmd(self, l, a):
        t = Twist(); t.linear.x, t.angular.z = float(l), float(a)
        self.pub.publish(t)

    def run(self):
        t0 = time.time()
        while time.time() - t0 < 40 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.y is None:
                continue
            if self.y < self.target_y:
                break
            # 保持朝北, 倒车(负前向)
            yaw_e = math.pi / 2 - self.yaw
            self.cmd(-0.18, max(-0.5, min(0.5, 1.5 * yaw_e)))
        self.cmd(0.0, 0.0)
        rclpy.spin_once(self, timeout_sec=0.3)
        print(f"[BACKOUT] arrived y={self.y:.3f} (target {self.target_y})")


if __name__ == "__main__":
    rclpy.init()
    n = BackOut(2.30)
    n.run()
    n.destroy_node()
    rclpy.shutdown()