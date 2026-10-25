import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

layout = json.load(open("/workspace/baseline/examples/supermarket_sorting/retail_competition_layout.json"))
slots = layout


class Check(Node):
    def __init__(self):
        super().__init__("yolo_check")
        self.dets = []
        self.create_subscription(Detection3DArray, "/kele/detections", self.cb, 10)

    def cb(self, msg):
        for d in msg.detections:
            if not d.results:
                continue
            cls = d.results[0].hypothesis.class_id
            p = d.results[0].pose.pose.position
            self.dets.append((cls, [p.x, p.y, p.z], d.header.stamp.sec))


def main():
    rclpy.init()
    n = Check()
    t0 = time.time()
    while time.time() - t0 < 12 and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node()
    rclpy.shutdown()

    print(f"collected {len(n.dets)} detections")
    pos_errs = []
    wrong = []
    for cls, pw, _ in n.dets:
        pw = np.array(pw)
        # nearest layout product (fixed baseline layout -> products at GT positions)
        best, best_d = None, 1e9
        for s in slots:
            d = np.linalg.norm(pw - np.array(s["world_position"]))
            if d < best_d:
                best_d, best = d, s
        pos_errs.append(best_d)
        if cls != best["object_kind"]:
            wrong.append((cls, best["object_kind"], np.round(best_d, 3)))
    if pos_errs:
        pos_errs = np.array(pos_errs)
        print(f"position error vs GT: mean={pos_errs.mean()*1000:.0f}mm "
              f"median={np.median(pos_errs)*1000:.0f}mm max={pos_errs.max()*1000:.0f}mm")
        print(f"category accuracy: {(1-len(wrong)/len(pos_errs))*100:.1f}%  "
              f"({len(pos_errs)-len(wrong)}/{len(pos_errs)})")
        if wrong:
            print("wrong:", wrong[:8])


if __name__ == "__main__":
    main()