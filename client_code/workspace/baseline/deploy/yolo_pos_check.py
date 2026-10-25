import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

layout = json.load(open("/workspace/supermarket_sorting_task/examples/supermarket_sorting/retail_competition_layout.json"))


class Check(Node):
    def __init__(self):
        super().__init__("yolo_pos_check")
        self.dets = []
        self.create_subscription(Detection3DArray, "/kele/detections", self.cb, 10)

    def cb(self, msg):
        for d in msg.detections:
            if not d.results:
                continue
            cls = d.results[0].hypothesis.class_id
            p = d.results[0].pose.pose.position
            self.dets.append((cls, [p.x, p.y, p.z]))


def main():
    rclpy.init()
    n = Check()
    t0 = time.time()
    while time.time() - t0 < 105 and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node()
    rclpy.shutdown()

    per_kind = {}
    for cls, pw in n.dets:
        pw = np.array(pw)
        best, best_d = None, 1e9
        for s in layout:
            if s["object_kind"] != cls:
                continue
            d = np.linalg.norm(pw - np.array(s["world_position"]))
            if d < best_d:
                best_d, best = d, s
        if best_d > 0.5:
            continue  # 找不到同类商品(误检),跳过
        per_kind.setdefault(cls, []).append((best_d, pw, np.array(best["world_position"])))

    print("per-kind position error (mm): mean / median / max")
    all_e = []
    for cls, lst in sorted(per_kind.items()):
        e = np.array([x[0] for x in lst]) * 1000
        all_e.extend(e)
        print(f"  {cls:12s} n={len(lst):3d}  {e.mean():5.0f} / {np.median(e):5.0f} / {e.max():5.0f}")
    if all_e:
        all_e = np.array(all_e)
        print(f"TOTAL: n={len(all_e)} mean={all_e.mean():.0f}mm median={np.median(all_e):.0f}mm")

    # 分解误差:检测位置 vs 同类真值,看 x/y/z 分量
    print("\ncomponent error (det - GT, mm), first 15 matched:")
    shown = 0
    for cls, lst in per_kind.items():
        for e, pw, gt in lst:
            comp = (pw - gt) * 1000
            print(f"  {cls:10s} dx={comp[0]:+6.0f} dy={comp[1]:+6.0f} dz={comp[2]:+6.0f}")
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break


if __name__ == "__main__":
    main()