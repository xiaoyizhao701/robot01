#!/usr/bin/env python3
"""货架先验 + 搜索规划模块 (纯规划, 不控制机器人, 不发布 cmd_vel).

职责:
  - 读取 retail_competition_layout.json 作为唯一货架先验(格位几何 + aruco_id)
  - get_target_prior(kind) -> 目标货架/层/列 + 期望区域(仅先验, 不是精确坐标)
  - search_plan(prior)     -> 有序搜索步骤: 目标层(中心→左右邻) → 相邻层
                             每步给出: level / slide / head_pitch / lateral / region
  - in_region(pw, prior, level) -> 候选是否落在当前搜索区域(层一致 + 横向margin)
  - level_from_z(z)        -> 世界 z -> L1/L2/L3

注意:
  - world_position 只作为"货架格位的空间先验", 商品允许偏移/斜放/跨格/被遮挡。
  - 本模块不 import rclpy, 不接触任何 ROS 话题。
"""
import json
import math

import numpy as np

LEVELS = ["L1", "L2", "L3"]

# 训练/实测验证过的三层观察参数 (与 grasp_config.LEVEL_SLIDE / LEVEL_HEAD_PITCH 一致)
#   slide 单位: m (0..0.87); head_pitch 单位: rad (-1.18..0.16)
LEVEL_POSES = {
    "L1": (0.30, -1.00),
    "L2": (0.15, -0.70),
    "L3": (0.00, -0.40),
}

# 货架前观察车道: 太近(2.8+)只能看到货架边框; 2.4~2.5 能看到整排(训练数据同分布)
SEARCH_Y = 2.45

# 层的世界 z 参考值(来自布局), 用于从检测世界坐标判层
LEVEL_Z = {"L1": 0.55, "L2": 0.90, "L3": 1.22}


class RackMap:
    def __init__(self, layout_path):
        with open(layout_path) as f:
            self.slots = json.load(f)

    # ---- 先验查询 ----
    def slots_of_kind(self, kind):
        return [s for s in self.slots if s.get("object_kind") == kind]

    def get_target_prior(self, kind, slot=None):
        """返回目标先验(货架/层/列/区域中心). slot 可指定; 默认该类第一个格位。"""
        slots = self.slots_of_kind(kind)
        if not slots:
            return None
        s = slot or slots[0]
        return {
            "kind": kind,
            "shelf": s["shelf"],
            "level": s["level"],
            "column": s["column"],
            "aruco_id": s.get("aruco_id"),
            "center": np.array(s["world_position"], dtype=float),
            "body": s.get("body"),
        }

    def level_from_z(self, z):
        """世界 z -> 最近的层。"""
        best, bd = "L2", 1e9
        for lv, zz in LEVEL_Z.items():
            d = abs(z - zz)
            if d < bd:
                bd, best = d, lv
        return best

    # ---- 搜索计划 ----
    def search_plan(self, prior, lateral_step=0.22, max_attempts=9):
        """有序搜索步骤: 先目标层(中心→左右邻), 再相邻层。

        每步: {level, slide, pitch, lateral, region_center_x, attempt}
        lateral 是相对目标格位中心的横向偏移(沿货架 x 方向), 允许商品跨格/偏放。
        """
        lv = prior["level"]
        order = [lv] + [x for x in LEVELS if x != lv]
        steps = []
        for i, level in enumerate(order):
            slide, pitch = LEVEL_POSES[level]
            offs = [0.0, lateral_step, -lateral_step] if i == 0 else [0.0]
            for off in offs:
                steps.append({
                    "level": level,
                    "slide": slide,
                    "pitch": pitch,
                    "lateral": off,
                    "region_center_x": float(prior["center"][0]) + off,
                    "attempt": len(steps) + 1,
                })
                if len(steps) >= max_attempts:
                    return steps
        return steps

    # ---- 区域判定 ----
    def in_region(self, p_world, prior, level, lateral_margin=0.45, level_tol=0.20):
        """候选是否在"目标层 + 货架横向范围"内(允许商品偏移/跨格/斜放)。

        不做小 ROI 卡死: 横向允许 ±lateral_margin(约±2个格位), 层允许 ±level_tol。
        """
        if p_world is None:
            return False
        z_ok = abs(float(p_world[2]) - LEVEL_Z.get(level, p_world[2])) <= level_tol
        if not z_ok:
            return False
        dx = abs(float(p_world[0]) - float(prior["center"][0]))
        return dx <= lateral_margin


if __name__ == "__main__":
    # 自检: 打印 sanmingzhi 的先验与搜索计划
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/workspace/baseline/examples/supermarket_sorting/retail_competition_layout.json"
    rm = RackMap(path)
    for kind in ("sanmingzhi", "maidong", "kele"):
        prior = rm.get_target_prior(kind)
        print(f"{kind}: shelf={prior['shelf']} level={prior['level']} "
              f"col={prior['column']} center={np.round(prior['center'], 2)}")
        for st in rm.search_plan(prior):
            print("   ", st)