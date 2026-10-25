#!/usr/bin/env python3
"""Drive the base to the shelf lane (x,y) facing north, then hold. For YOLO demo."""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Drive(Node):
    def __init__(self, tx, ty, tyaw, hold=30.0):
        super().__init__("drive_demo")
        self.tx, self.ty, self.tyaw = tx, ty, tyaw
        self.hold = hold
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
        parked = False
        park_t = 0.0
        while time.time() - t0 < 90 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.x is None:
                continue
            dx, dy = self.tx - self.x, self.ty - self.y
            dist = math.hypot(dx, dy)
            yaw_e = wrap(self.tyaw - self.yaw)
            if not parked:
                if dist < 0.10 and abs(yaw_e) < 0.06:
                    parked = True
                    park_t = time.time()
                    print("[drive] parked at", round(self.x, 2), round(self.y, 2), flush=True)
                    continue
                lin = min(0.35, 1.2 * dist) * max(0.25, math.cos(wrap(math.atan2(dy, dx) - self.yaw)))
                self.cmd(max(-0.30, min(0.30, lin)), max(-0.8, min(0.8, 1.6 * yaw_e)))
            else:
                self.cmd(0.0, max(-0.5, min(0.5, 2.0 * yaw_e)))
                if time.time() - park_t > self.hold:
                    break
        self.cmd(0.0, 0.0)
        print("[drive] done, parked at", round(self.x, 2), round(self.y, 2), flush=True)


if __name__ == "__main__":
    rclpy.init()
    d = Drive(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
              float(sys.argv[4]) if len(sys.argv) > 4 else 30.0)
    d.run()
    d.destroy_node()
    rclpy.shutdown()