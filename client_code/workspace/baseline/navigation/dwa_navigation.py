"""External navigation layer for the supermarket client.

Everything about *how the robot navigates* lives here so it can be tuned in
isolation: the obstacle-zone gating, the DWA planner hookup, waypoint advance,
and the converge/pair-off tweaks. Consumers mix `DwaNavigationMixin` into a
client that provides the base pose/scan/twist plumbing.

The client contract this mixin relies on (all provided by the base client):
  - self.base_xy (odom x,y), self.base_yaw
  - self.set_twist(lin, ang), self.cur_lin, self.cur_ang
  - self.nav_idx, self.nav_mode, self.pos_tol, self.turn_tol
  - self.reset_nav() and a `follow_route` fallback (plain straight route)
"""
import os
import math

import numpy as np
import rclpy
from sensor_msgs.msg import LaserScan

from dwa_planner import DWAPlanner
from astar_planner import AstarPlanner
from supermarket_sorting_client import wrap_to_pi

# ---- navigation tuning (edit here, no need to touch the client) ----
# dynamic_obstacle_corridor world-frame bounds (corridor pos "-0.96 -1.01 0").
OBSTACLE_ZONE = dict(x_min=-2.5, x_max=0.5, y_min=-3.75, y_max=1.70)

# Route to the delivery desk (world frame). When the robot starts at the spawn
# lane (1.92,-3.15) and skips the grasp, it must NOT cut straight to (-1.2,2.3):
# the corridor_right_board wall sits at world x=0.53 and blocks y in [-3.7,1.7],
# so the first leg drives north FIRST (east of that wall), then west at y=2.3
# (above the wall top), then south through the box corridor with DWA.
ROUTE_TO_TABLE = [
    [1.52, 2.30],
    [-1.20, 2.30],
    [-1.20, -0.70],
    [-1.20, -2.80],
    [-1.88, -2.80],
]

ZONE_PADDED = 0.0          # expand the zone by this much on every side (m)
ADVANCE_BAND = 2.0         # only skip a waypoint via plane-crossing once the
                           # robot is this close to the current route segment,
                           # otherwise it skips early on perpendicular legs
CONVERGE_RAD = 0.6         # switch to pure bearing-PD settle inside this (m)
FINAL_YAW_GAIN = 1.8       # last-waypoint heading alignment gain
PASS_OVERLAP = 0.0         # positive < 0 -> advance waypoint before crossing plane

# Shelf approach lane: the robot drives along y≈2.475 (yellow picking zone mid)
# to reach each shelf. A lane-hold keeps it on that line so it never drifts north
# into the shelf row (which wedged it at y≈3.06).
SHELF_LANE_Y = 2.50

# heading-hold gains for the converge leg (bearing PD, not DWA).
CONVERGE_LIN_MAX = 0.35
CONVERGE_LIN_GAIN = 0.45
CONVERGE_ANG_GAIN = 1.4
CONVERGE_ANG_DAMP = 0.4


class DwaNavigationMixin:
    """Zone-gated DWA navigation: fast plain route following outside the
    obstacle zone, lidar avoidance through the box corridor inside it."""

    # stuck-recovery constants
    STUCK_THRESH = 0.04          # position change below this (m) → stuck
    STUCK_TICKS = 400            # ticks without progress before recovery (~8s)
    RECOVERY_TICKS = 50          # reverse duration during recovery (~2.5s)
    RECOVERY_LIN = -0.3          # reverse speed during recovery
    MAX_RECOVERIES = 4           # give up after this many recovery attempts

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Slower DWA caps for a steadier ride with the arm extended (v_max=1.3
        # was twitchy and the robot tipped). Matches the client's 0.45 m/s cap.
        # robot_radius 0.30->0.25: 箱阵缝隙按 0.347 校验, 0.30+0.03 间隙下可操作
        # 余量只剩几厘米导致 DWA 蠕动(走廊 60-100s); 0.25 仍大于实际半宽 0.22。
        self.planner = DWAPlanner(v_max=0.75, w_max=0.90, a_v=0.8, a_w=2.5,
                                  robot_radius=float(os.getenv("DWA_RADIUS", "0.25")),
                                  horizon=float(os.getenv("DWA_HORIZON", "2.5")))
        # A* 膨胀 0.35 = 校验器极限, 走廊通道(0.69m)在图上恰好为 0 宽 -> A*
        # 经常找不到路; 0.28+0.03=0.31 让路径稳定存在(实际半径 0.22 仍有余量)。
        self.astar = AstarPlanner(self.planner, robot_radius=0.28, clearance=0.03)
        self.converge_rad = CONVERGE_RAD
        self.scan = None
        self._stuck_count = 0
        self._stuck_anchor = None
        self._recovery_ticks = 0
        self._recovery_ang = 0.0  # angular component during recovery
        self._last_pos = None
        self._pos_unchanged_ticks = 0
        self.create_subscription(
            LaserScan, "/slamware_ros_sdk_server_node/scan", self.scan_cb, 5)
        self.get_logger().info("DWA obstacle avoidance enabled (zone gated)")

    def scan_cb(self, msg):
        self.scan = msg

    def in_obstacle_zone(self):
        if self.base_xy is None:
            return False
        x, y = self.base_xy
        pad = ZONE_PADDED
        z = OBSTACLE_ZONE
        return (z["x_min"] - pad <= x <= z["x_max"] + pad) and \
               (z["y_min"] - pad <= y <= z["y_max"] + pad)

    def follow_route(self, route, final_yaw):
        """Follow `route` waypoint-by-waypoint to its final point (route[-1]).

        Waypoint following (not a straight shot to the end) is what keeps the
        robot on the intended lane: from the spawn it first drives NORTH along
        x=1.92 (east of the corridor_right_board wall at world x=0.53, which
        blocks y in [-3.7,1.7]), then WEST at y=2.30/2.475 above the wall top,
        then south through the box corridor. Inside the obstacle zone we run
        A*+DWA for box/lidar avoidance; in the open shelf/table areas we drive
        straight to the current waypoint and align the heading at the final
        point. This avoids both the nav_idx desync freeze and the straight-line
        wall collision."""
        if self.base_xy is None:
            return super().follow_route(route, final_yaw)

        # ---- stuck detection & recovery (obstacle zone only) ----
        # On the open start->shelf and shelf->table approaches there are no
        # obstacles, so a "stuck" reading is spurious (or a slow-align moment);
        # the reverse+rotate recovery there only served to flip the robot over.
        # Only run it inside the box corridor where a genuine wedge can happen.
        if self.in_obstacle_zone():
            if self._recovery_ticks > 0:
                # BIG 逃逸: 倒到周围开阔(近障 >0.5m)就提前结束, 不必固定时长;
                # 普通回滚按原时长。
                if getattr(self, "_big_escape", False) and self.scan is not None:
                    _r = np.asarray(self.scan.ranges, dtype=float)
                    _ok = np.isfinite(_r) & (_r > 0.05)
                    if _ok.any() and float(np.min(_r[_ok])) > 0.5:
                        self._recovery_ticks = 0
                        self._big_escape = False
                        return False
                self._recovery_ticks -= 1
                self.set_twist(self.RECOVERY_LIN, self._recovery_ang)
                return False
            cur = tuple(self.base_xy)
            if self._last_pos is not None:
                window_moved = math.hypot(cur[0] - self._last_pos[0],
                                          cur[1] - self._last_pos[1])
                # POSITION ONLY: spinning in place (yaw changing, position not)
                # is the failure we must break, so yaw no longer counts as
                # progress. A longer window (STUCK_TICKS ~8s) keeps the brief
                # in-place align at the shelf from being flagged.
                if window_moved >= self.STUCK_THRESH:
                    self._pos_unchanged_ticks = 0
                    # 只有真正离开上次卡死点 1.5m 才清零回滚计数: 否则
                    # "回滚->前进->再卡"循环永远从 #1 开始, 力度永远不够,
                    # 会一直重复同一个错误方案(用户反馈)。
                    if (self._stuck_anchor is None
                            or math.hypot(cur[0] - self._stuck_anchor[0],
                                          cur[1] - self._stuck_anchor[1]) > 1.5):
                        self._stuck_count = 0
                        self._stuck_anchor = None
                    self._last_pos = cur
                else:
                    self._pos_unchanged_ticks += 1
            else:
                self._last_pos = cur
            if self._pos_unchanged_ticks >= self.STUCK_TICKS:
                self._pos_unchanged_ticks = 0
                self._stuck_count += 1
                if self._stuck_anchor is None:
                    self._stuck_anchor = cur
                self._last_pos = cur
                if self._stuck_count <= self.MAX_RECOVERIES:
                    sign = 1.0 if (self._stuck_count % 2 == 0) else -1.0
                    self._recovery_ang = sign * 1.5
                    # 回滚力度随次数递增(回滚不够会一直重复同一个错误方案),
                    # 并清掉 A* 缓存强制重新规划(旧路径可能就是错的)。
                    self._recovery_ticks = self.RECOVERY_TICKS + 30 * self._stuck_count
                    try:
                        self.astar._path = None
                    except Exception:
                        pass
                    self.get_logger().warn(
                        "stuck (#%d) — reverse+rotate sign=%+.0f ticks=%d" %
                        (self._stuck_count, sign, self._recovery_ticks))
                    self.set_twist(self.RECOVERY_LIN, self._recovery_ang)
                    return False
                else:
                    # BIG 逃逸次数: 连续 3 次仍卡死说明方案根本不对(膨胀区/
                    # 局部地图问题), 不再硬耗 -> 置放弃标志, 客户端退出 rc=1,
                    # 调度器重置场景跑下一单。
                    self._big_count = getattr(self, "_big_count", 0) + 1
                    if self._big_count >= 3:
                        self.get_logger().error(
                            "BIG escape x%d still stuck; aborting this order" %
                            self._big_count)
                        self.nav_abort = True
                        self.set_twist(0.0, 0.0)
                        return False
                    sign = 1.0 if (self._stuck_count % 2 == 0) else -1.0
                    self.get_logger().warn(
                        "stuck recovery exhausted — BIG reverse escape sign=%+.0f" % sign)
                    self._recovery_ang = sign * 1.2
                    self._big_escape = True
                    self._recovery_ticks = 100          # ~5s of reversing (early-exit when clear)
                    try:
                        self.astar._path = None
                    except Exception:
                        pass
                    self._stuck_count = 0
                    self.set_twist(-0.40, self._recovery_ang)
                    return False
        else:
            self._pos_unchanged_ticks = 0
            self._recovery_ticks = 0
            self._stuck_count = 0

        route = [tuple((float(w[0]), float(w[1]))) for w in route]
        n = len(route)
        if self.nav_idx >= n or self.nav_idx < 0:
            self.nav_idx = 0
        # 过平面推进(用户建议"看 y 值"): 只要沿"当前段->下一段"方向越过了
        # 当前路径点的垂面(带容差), 或者已经贴近, 就算过 —— 绕远/超调时不必
        # 折返回到那个点(旧逻辑 dist<0.15 太刚性, 卡在障碍间反复回头)。
        while self.nav_idx < n - 1:
            wp = route[self.nav_idx]
            nxt = route[self.nav_idx + 1]
            seg_x = nxt[0] - wp[0]
            seg_y = nxt[1] - wp[1]
            seg_len = math.hypot(seg_x, seg_y)
            d_wp = math.hypot(self.base_xy[0] - wp[0], self.base_xy[1] - wp[1])
            # 推进阈值 >= DWA 的 goal_tolerance(0.22): 否则 DWA 在
            # 路径点 0.13~0.22m 处判定"到达"就停住, 而路径点推进(<0.15)
            # 不触发 -> 永久卡死(zhijin 返程卡死的根因)。
            if d_wp < max(self.pos_tol, 0.30):
                self.nav_idx += 1
                continue
            if seg_len < 1e-6:
                break
            # 脚本路径(固定折线)必须逐点走, 不做"过平面"跳过: 入口点距机器人
            # 0.95m 时 proj 达 0.93 会被误跳过 -> 机器人朝错误方向撞墙(西墙事故)。
            if getattr(self, "scripted_path", False):
                break
            ux, uy = seg_x / seg_len, seg_y / seg_len
            proj = (self.base_xy[0] - wp[0]) * ux + (self.base_xy[1] - wp[1]) * uy
            if proj > 0.30 and d_wp < 3.5:
                self.get_logger().info(
                    f"[nav] waypoint {self.nav_idx} passed by plane "
                    f"(proj={proj:.2f} d={d_wp:.2f})")
                self.nav_idx += 1
                continue
            break
        target = route[self.nav_idx]
        dx = target[0] - self.base_xy[0]
        dy = target[1] - self.base_xy[1]
        dist = math.hypot(dx, dy)

        final = (self.nav_idx == n - 1)

        # Arrived at the final point: align heading to final_yaw.
        if final and dist < self.pos_tol:
            yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
            self.set_twist(0.0, FINAL_YAW_GAIN * yaw_err)
            if abs(yaw_err) < self.turn_tol:
                self.set_twist(0.0, 0.0)
                return True
            return False

# Open area (shelf/table sides) or no lidar yet: drive straight to the
        # current waypoint; align only at the final point.
        if not self.in_obstacle_zone() or self.scan is None:
            # On the shelf side (north of the corridor, near the shelf row), hold
            # the approach lane so the robot can't drift north into the shelves.
            if (target[1] > 2.0 and 1.7 < self.base_xy[1] < 3.1
                    and self.nav_idx >= n - 2):
                return self._shelf_lane_hold(target, SHELF_LANE_Y, final_yaw, final)
            return self._drive_to(target, final_yaw if final else None,
                                  align_final=final)

        # 固定通道路径(scripted): 预设折线只当"引导"(目标点), 局部避障
        # 交给 DWA(含稳定代价), 不再纯 bearing 直线硬走(用户: 路线太固定)。
        # 脚本点间距 ~1.2m, DWA 只需够到下一个点, 不会陷进全局局部极小。
        if getattr(self, "scripted_path", False) and self.in_obstacle_zone():
            pose_s = (self.base_xy[0], self.base_xy[1], self.base_yaw)
            lin, ang = self.planner.plan(
                pose_s, target,
                self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
                v_now=self.cur_lin, w_now=self.cur_ang, follow_path=False,
                radius=0.24,
            )
            lin = min(lin, 0.16)
            ang = float(np.clip(ang, -0.55, 0.55))
            self.set_twist(lin, ang)
            return False

        # Inside the obstacle zone: close-in bearing-PD settle, else A*+DWA.
        if dist < self.converge_rad:
            brg = math.atan2(dy, dx)
            head_err = wrap_to_pi(brg - self.base_yaw)
            align = max(0.15, math.cos(head_err))
            lin = min(CONVERGE_LIN_MAX, CONVERGE_LIN_GAIN * dist * align)
            ang = CONVERGE_ANG_GAIN * head_err - CONVERGE_ANG_DAMP * self.cur_ang
            self.set_twist(lin, ang)
            return False

        pose = (self.base_xy[0], self.base_xy[1], self.base_yaw)
        # 纯跟踪(pure pursuit)沿 A* 路径过箱阵: DWA 在 0.35m 级缝隙里
        # 蠕动(0.05-0.1 m/s, 5.5m 走廊要 60-110s 且方差大), 而 A* 路径本身
        # 已按 0.35m 间隙校验过, 直接跟踪可稳定在 ~20s。
        # 只在离路径点 >0.7m 时用(终点收敛用 DWA; 0.7-1.2m 恰是箱阵窄缝,
        # DWA 在那里找不到路 -> 卡死回滚, 交给纯跟踪沿 A* 路径绕)。
        if dist < 0.7:
            return self._dwa_step(pose, target)
        # 近障感知: 距离越近, 前瞻越短/速度越低/加反向偏置, 防止切角进入
        # 膨胀障碍区(进去后 A* 无路 -> 卡死)。
        scan_min = 9.9
        obs_ang = 0.0
        if self.scan is not None:
            _rr = np.asarray(self.scan.ranges, dtype=float)
            _aa = self.scan.angle_min + np.arange(len(_rr)) * self.scan.angle_increment
            _cone = (np.abs(_aa) < 1.1) & np.isfinite(_rr)
            if _cone.any():
                _rc = np.where(_cone, _rr, np.inf)
                _i = int(np.argmin(_rc))
                scan_min = float(_rc[_i])
                obs_ang = float(_aa[_i])
        look = 0.5 if scan_min >= 0.45 else 0.25
        tgt = self.astar.lookahead_target((pose[0], pose[1]), target, lookahead=look)
        if tgt is None:
            # A* 无路(机器人被膨胀障碍包围, 常见于切角进缝后): 不能用
            # "直线冲向路径点"兜底(会顶着箱子卡死), 交给 DWA 倒车+局部避障。
            return self._dwa_step(pose, target)
        dx_t = tgt[0] - pose[0]
        dy_t = tgt[1] - pose[1]
        dist_t = math.hypot(dx_t, dy_t)
        if dist_t < 1e-3:
            return self._dwa_step(pose, target)
        brg_t = math.atan2(dy_t, dx_t)
        head_err_t = wrap_to_pi(brg_t - pose[2])
        align = max(0.25, math.cos(head_err_t))
        lin = min(0.16, 0.9 * dist_t) * align
        if scan_min < 0.45:
            lin = min(lin, 0.15)
        if scan_min < 0.30:
            lin = min(lin, 0.08)
        if scan_min < 0.20:
            lin = 0.0
        ang = float(np.clip(1.3 * head_err_t, -0.55, 0.55))
        # 避障偏置: 近障(<=0.45)时朝障碍反方向偏(越近越强)
        if scan_min < 0.45 and scan_min < 9.0:
            bias = -1.6 * (0.45 - scan_min) / 0.45 * (1.0 if obs_ang >= 0 else -1.0)
            ang += bias
        self.set_twist(lin, ang)
        return False

    def _dwa_step(self, pose, target):
        tgt = target
        lin, ang = self.planner.plan(
            pose, tuple(tgt),
            self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
            v_now=self.cur_lin, w_now=self.cur_ang, follow_path=True,
        )
        self.set_twist(lin, ang)
        return False

    def seed_known_obstacles(self, boxes_world):
        """把已知静态障碍(箱阵)直接写进 DWA 世界占据栅格。

        lidar 逐帧建图导致"先往窄缝走 -> 到了才发现过不去 -> 等地图更新再绕",
        走廊 choke 点浪费 ~50s。箱阵是静态且按种子已知的, 直接预置即可。
        boxes_world: [(cx, cy, yaw, sx, sy), ...] 世界坐标+尺寸(m)。
        """
        m = self.planner
        if m.map_origin is None:
            return False
        ox, oy = m.map_origin
        res = m.map_res
        n = m.map_n
        infl = float(os.getenv("SEED_INFL", "0.30"))  # 0.22 试过更差(卡死+翻车), 回滚 0.30
        for (cx, cy, yaw, sx, sy) in boxes_world:
            _infl = infl
            if len(boxes_world[0]) >= 6 and sx == 0.1:
                _infl = 0.05   # 墙: 只给极小膨胀(防 A* 穿墙), 不挤压 DWA 通道
            hx = sx * 0.5 + _infl
            hy = sy * 0.5 + _infl
            c, s = math.cos(yaw), math.sin(yaw)
            r = math.hypot(hx, hy) + res
            ix0 = max(0, int((cx - r - ox) / res))
            ix1 = min(n - 1, int((cx + r - ox) / res))
            iy0 = max(0, int((cy - r - oy) / res))
            iy1 = min(n - 1, int((cy + r - oy) / res))
            for iy in range(iy0, iy1 + 1):
                wy = oy + iy * res
                for ix in range(ix0, ix1 + 1):
                    wx = ox + ix * res
                    dx = wx - cx
                    dy = wy - cy
                    lx = c * dx + s * dy
                    ly = -s * dx + c * dy
                    if abs(lx) <= hx and abs(ly) <= hy:
                        m.map[iy, ix] = 100
        m.map_ver += 1
        return True

    def _shelf_lane_hold(self, target_xy, lane_y, final_yaw, final=False):
        """Drive along the shelf approach lane (y≈lane_y) toward target_xy,
        keeping the base strictly on the lane so it can't drift north into the
        shelf row. On the FINAL waypoint, once near the slot, rotate to face the
        shelf (final_yaw) so the head camera sees the product; only then return
        True (otherwise the robot faced west and DEPLOY saw 0 detections)."""
        dx = target_xy[0] - self.base_xy[0]
        dy_t = target_xy[1] - self.base_xy[1]
        near = math.hypot(dx, dy_t) < 0.12

        if final and near:
            # arrived at the slot: turn to face the shelf, then done.
            yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
            self.set_twist(0.0, FINAL_YAW_GAIN * yaw_err)
            return abs(yaw_err) < self.turn_tol

        # steer straight to the target point (its own x AND y). Aiming at a
        # forced lane y or the robot's own x made the bearing undefined and spun
        # the robot in place.
        gx, gy = target_xy[0], target_xy[1]
        brg = math.atan2(gy - self.base_xy[1], gx - self.base_xy[0])
        head_err = wrap_to_pi(brg - self.base_yaw)
        dist = math.hypot(gx - self.base_xy[0], gy - self.base_xy[1])
        lin = min(float(os.getenv("LANE_LIN_MAX", "0.30")), 0.5 * dist)
        ang = 1.8 * head_err - 0.5 * self.cur_ang
        self.set_twist(lin, ang)
        return False

    def _drive_to(self, target_xy, final_yaw=None, align_final=False):
        """Track `target_xy` with DWA (in obstacle zone) / bearing (open area)."""
        pose = (self.base_xy[0], self.base_xy[1], self.base_yaw)
        delta = (target_xy[0] - pose[0], target_xy[1] - pose[1])
        dist = math.hypot(*delta)

        if align_final and dist < self.pos_tol:
            yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
            self.set_twist(0.0, FINAL_YAW_GAIN * yaw_err)
            if abs(yaw_err) < self.turn_tol:
                self.set_twist(0.0, 0.0)
                return True
            return False

        # Close to the goal: settle with a bearing PD so we don't overshoot.
        if dist < self.converge_rad:
            brg = math.atan2(delta[1], delta[0])
            head_err = wrap_to_pi(brg - self.base_yaw)
            align = max(0.15, math.cos(head_err))
            lin = min(CONVERGE_LIN_MAX, CONVERGE_LIN_GAIN * dist * align)
            ang = CONVERGE_ANG_GAIN * head_err - CONVERGE_ANG_DAMP * self.cur_ang
            self.set_twist(lin, ang)
            return False

        # Far away: DWA (with A* look-ahead) inside the obstacle zone, otherwise a
        # direct bearing drive through the open area.
        if self.in_obstacle_zone() and self.scan is not None:
            target = self.astar.lookahead_target((pose[0], pose[1]), target_xy)
            if target is None:
                target = tuple(target_xy)
            lin, ang = self.planner.plan(
                pose, tuple(target),
                self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
                v_now=self.cur_lin, w_now=self.cur_ang, follow_path=True,
            )
            self.set_twist(lin, ang)
            return False

        brg = math.atan2(delta[1], delta[0])
        head_err = wrap_to_pi(brg - self.base_yaw)
        abs_err = abs(head_err)
        # Arc-drive: always give some forward motion (even when misaligned)
        # to avoid spin-in-place wedging near the shelf.  Speed drops as
        # heading error grows but never reaches zero.
        align = max(0.15, math.cos(head_err))
        lin = min(1.0, 1.0 * dist * align)
        if abs_err > 2.5:
            # Very large error: slow forward + strong turn (arc, not spin)
            lin = min(lin, 0.25)
        elif abs_err > math.pi / 2:
            # Moderate error: reduced forward + steering
            lin = min(lin, 0.5)
        ang = 2.2 * head_err - 0.6 * self.cur_ang
        self.set_twist(lin, ang)
        return False