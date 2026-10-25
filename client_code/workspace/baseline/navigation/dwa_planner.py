"""Pure-Python Dynamic Window Approach (DWA) local planner.

No Nav2 / SLAM required. Consumes a single-line LaserScan in the robot frame,
builds obstacles, and produces a (linear, angular) twist away from them and
toward a goal. Designed to run mounted inside the supermarket sorting client
container, which only needs numpy / scipy + ROS2 Humble.

In addition to a classic DWA dynamic window the planner implements three
behaviors borrowed from Nav2's costmap + recovery stack:

  * occupancy map : a world-frame grid is accumulated from every lidar scan
    (ray-casted free cells + occupied hits) and fused back into the local
    costmap, so a box seen earlier still steers the robot away from it later.
  * recovery      : when the robot stalls in a local minimum it first rotates
    toward the widest free lidar gap; if that makes no progress it reverses
    out (Nav2's rotate/backup recovery equivalents).
  * adaptive speed: the forward cone's free distance scales v_max, so the
    robot cruises fast in open aisles and slows only near obstacles.

The MMK2 AGV footprint is roughly a 0.44 x 0.40 m box centered on base_link
(including the wheels). The server's obstacle validator guarantees a route at
ROBOT_CLEARANCE_RADIUS = hypot(0.22, 0.20) + 0.05 = 0.347 m, and the gaps
between the random boxes are only just wider than that, so the planner's
collision radius matches 0.347 with a small inflation margin.
"""
import os
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt


class DWAPlanner:
    """Dynamic-window local planner.

    pose          : (x, y, yaw) in a world frame (odom)
    goal          : (x, y) in the same world frame
    scan_ranges   : 1-D float array from a LaserScan (meter), inf where no return
    """

    def __init__(
        self,
        robot_radius=0.30,
        inflation=0.03,
        v_max=1.30,
        w_max=1.10,
        a_v=1.5,
        a_w=6.0,
        dt=0.1,
        horizon=2.5,
        goal_tolerance=0.22,
        laser_offset=0.114,
        grid_half=3.0,
        res=0.05,
        obs_w=2.0,
        goal_pot=float(os.getenv("DWA_GOAL_POT", "8.0")),
        vel_w=0.2,
        rotate_threshold=0.7,
        rotate_gain=2.0,
        v_min=0.22,
        speed_clear=1.3,
        map_half=7.0,
        map_res=0.05,
    ):
        self.robot_radius = robot_radius
        self.inflation = inflation
        self.clearance = robot_radius + inflation
        self.v_max = v_max
        self.v_min = v_min
        self.speed_clear = speed_clear
        self.w_max = w_max
        self.a_v = a_v
        self.a_w = a_w
        self.dt = dt
        self.horizon = horizon
        self.n_steps = max(4, int(horizon / dt))
        self.goal_tolerance = goal_tolerance
        self.laser_offset = laser_offset
        self.grid_half = grid_half
        self.res = res
        self.obs_w = obs_w
        self.goal_pot = goal_pot
        self.vel_w = vel_w
        self.rotate_threshold = rotate_threshold
        self.rotate_gain = rotate_gain

        # world-frame occupancy grid (odom). -1 unknown, 0 free, 100 occupied.
        self.map_half = map_half
        self.map_res = map_res
        self.map_n = int(2 * map_half / map_res)
        self.map = np.full((self.map_n, self.map_n), -1, dtype=np.int8)
        self.map_origin = None
        self.map_ver = 0          # bumped every time the world map is updated
        self._edt_main = None     # cached inflated-distance grid (for safe_goal)

        # stall / recovery state
        self._stall_t = None
        self._stall_xy = None
        self._reverse_t = None
        self._recover_mode = None     # None | "align" | "drive"
        self._recover_gap = None      # locked gap bearing during a recovery run
        self._recover_t = 0.0
        self._last_map_t = 0.0
        self._map_period = 0.2   # accumulate lidar into the map at ~5 Hz
        self._last_dbg = 0.0     # debug-log throttle
        self._dbg_period = 1.0

    # ------------------------------------------------------------------ map
    def _update_map(self, x, y, yaw, ranges, angle_min, angle_increment):
        """Ray-cast the current scan into the world-frame occupancy grid."""
        if self.map_origin is None:
            self.map_origin = (x - self.map_half, y - self.map_half)
        ox, oy = self.map_origin
        n = self.map_n
        c, s = math.cos(yaw), math.sin(yaw)
        r = np.asarray(ranges, dtype=float)
        mask = np.isfinite(r) & (r > 0.05) & (r < self.map_half)
        if not mask.any():
            return
        self.map_ver += 1
        angles = angle_min + np.arange(len(r)) * angle_increment
        ra = r[mask]
        aa = angles[mask]
        # world hit points
        lx = ra * np.cos(aa) - self.laser_offset
        ly = ra * np.sin(aa)
        hx = x + c * lx - s * ly
        hy = y + s * lx + c * ly
        # robot cell + hit cells in map indices
        rx = int((x - ox) / self.map_res)
        ry = int((y - oy) / self.map_res)
        gx = ((hx - ox) / self.map_res).astype(int)
        gy = ((hy - oy) / self.map_res).astype(int)
        inside = (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
        gx, gy = gx[inside], gy[inside]
        # simple DDA over each ray to clear free cells
        for i in range(len(gx)):
            x0, y0 = rx, ry
            x1, y1 = int(gx[i]), int(gy[i])
            dx, dy = x1 - x0, y1 - y0
            steps = max(abs(dx), abs(dy), 1)
            for k in range(1, steps + 1):
                cx = x0 + int(round(dx * k / steps))
                cy = y0 + int(round(dy * k / steps))
                if 0 <= cx < n and 0 <= cy < n:
                    self.map[cy, cx] = 0
            # last cell is the occupied hit
            if 0 <= x1 < n and 0 <= y1 < n:
                self.map[y1, x1] = 100

    def seed_static(self, boxes):
        """Mark known-static world-frame rectangles (list of (x0, y0, x1, y1))
        as occupied in the occupancy grid. The A* global planner inflates these
        so it never routes through or *hugs* a wall the robot already knows is
        there (corridor boards, shelf row, perimeter, delivery table), which is
        exactly why it used to graze them and spin."""
        if self.map_origin is None:
            self.map_origin = (self.map_half * -1.0, self.map_half * -1.0)
        ox, oy = self.map_origin
        res = self.map_res
        n = self.map_n
        for (x0, y0, x1, y1) in boxes:
            gx0 = int((x0 - ox) / res)
            gy0 = int((y0 - oy) / res)
            gx1 = int((x1 - ox) / res)
            gy1 = int((y1 - oy) / res)
            gx0 = max(0, min(n - 1, gx0)); gx1 = max(0, min(n - 1, gx1))
            gy0 = max(0, min(n - 1, gy0)); gy1 = max(0, min(n - 1, gy1))
            if gx1 < gx0: gx0, gx1 = gx1, gx0
            if gy1 < gy0: gy0, gy1 = gy1, gy0
            self.map[gy0:gy1 + 1, gx0:gx1 + 1] = 100
        self.map_ver += 1
        self._edt_main = None

    def _map_occ_in_local(self, x, y, yaw, half, res):
        """Occupied map cells transformed into the base-fixed local frame."""
        if self.map_origin is None:
            return None
        ox, oy = self.map_origin
        occ = np.argwhere(self.map == 100)
        if not len(occ):
            return None
        mx = ox + (occ[:, 1] + 0.5) * self.map_res
        my = oy + (occ[:, 0] + 0.5) * self.map_res
        c, s = math.cos(yaw), math.sin(yaw)
        dx = mx - x
        dy = my - y
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        ok = (np.abs(bx) < half) & (np.abs(by) < half)
        if not ok.any():
            return None
        bx, by = bx[ok], by[ok]
        ix = ((bx + half) / res).astype(int)
        iy = ((by + half) / res).astype(int)
        n = int(2 * half / res)
        keep = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        return iy[keep], ix[keep]

    # ------------------------------------------------------------ safety
    def point_potential(self, gx, gy):
        """Return the inflate-distance (m) to the nearest obstacle at (gx, gy),
        or None when the world map has no origin (nothing scanned yet)."""
        if self.map_origin is None:
            return None
        ox, oy = self.map_origin
        res = self.map_res
        n = self.map_n
        ix = int((gx - ox) / res)
        iy = int((gy - oy) / res)
        if not (0 <= ix < n and 0 <= iy < n):
            return None
        if self._edt_main is None:
            self._edt_main = distance_transform_edt(~(self.map == 100)) * res
        return float(self._edt_main[iy, ix])

    def safe_goal(self, gx, gy, inflate=0.30):
        """Project a goal that lands inside/too-close-to an obstacle to the
        nearest free cell, so the controller never steers into a box (which
        makes it spin in place). Returns (x, y)."""
        if self.map_origin is None:
            return (gx, gy)
        ox, oy = self.map_origin
        res = self.map_res
        n = self.map_n
        free = self.map != 100
        if not free.any():
            return (gx, gy)
        if self._edt_main is None:
            self._edt_main = distance_transform_edt(~(self.map == 100)) * res
        ix = int((gx - ox) / res)
        iy = int((gy - oy) / res)
        # bounce the cell into range, then only nudge if it is too tight.
        if not (0 <= ix < n and 0 <= iy < n):
            ix = max(0, min(n - 1, ix))
            iy = max(0, min(n - 1, iy))
        if self._edt_main[iy, ix] >= inflate:
            return (gx, gy)
        # nearest free cell: search outward for the closest cell with clearance.
        best = None
        best_d = 1e9
        for r in range(1, 24):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    yy, xx = iy + dy, ix + dx
                    if not (0 <= xx < n and 0 <= yy < n):
                        continue
                    if self.map[yy, xx] == 100:
                        continue
                    d = math.hypot(xx - ix, yy - iy)
                    if d < best_d:
                        best_d = d
                        best = (xx, yy)
            if best is not None and best_d * res >= 1.5 * res:
                break
        if best is None:
            return (gx, gy)
        bx, by = best
        return (ox + (bx + 0.5) * res, oy + (by + 0.5) * res)

    # ------------------------------------------------------------ planning
    def plan(self, pose, goal, ranges, angle_min, angle_increment, v_now=0.0,
             w_now=0.0, follow_path=False, radius=None):
        # 调用方可收紧碰撞半径(脚本路径已验证安全, 用实际半宽 0.24 让
        # DWA 能穿过 0.69m 窄通道——默认 0.25+0.03 clearance 会把
        # 可走中线压到 ±0.11m, 轨迹几乎全被拒 -> 走廊卡死)。
        if radius is not None:
            self._tmp_radius = radius
        else:
            self._tmp_radius = self.robot_radius
        """Return (linear, angular) twist toward goal avoiding obstacles.

        With ``follow_path=True`` the goal is a nearby look-ahead point on a
        global A* path: it is always reachable and collision-checked, so the
        stop-and-turn branch is disabled (the robot arcs around corners) and a
        momentarily blocked goal direction is never treated as a stall - the
        planner relies on trajectory cost + recovery for obstacles instead.
        """
        x, y, yaw = pose
        gx, gy = goal

        c = math.cos(yaw)
        s = math.sin(yaw)

        # accumulate the world occupancy map from this scan (throttled to 5 Hz)
        now = time.monotonic()
        if now - self._last_map_t >= self._map_period:
            self._update_map(x, y, yaw, ranges, angle_min, angle_increment)
            self._last_map_t = now

        pts = self._scan_points(ranges, angle_min, angle_increment)

        # local costmap in a base-fixed frame (robot at origin, +x forward)
        half = self.grid_half
        res = self.res
        n = int(2 * half / res)

        grid = np.zeros((n, n), dtype=bool)
        if pts is not None and len(pts):
            bx = pts[:, 0] - self.laser_offset
            by = pts[:, 1]
            ix = ((bx + half) / res).astype(int)
            iy = ((by + half) / res).astype(int)
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
            grid[iy[ok], ix[ok]] = True

        # fuse remembered occupied cells from the world map
        mem = self._map_occ_in_local(x, y, yaw, half, res)
        if mem is not None:
            grid[mem[0], mem[1]] = True

        dist = distance_transform_edt(~grid) * res

        # goal in the local base frame
        relx = c * (gx - x) + s * (gy - y)
        rely = -s * (gx - x) + c * (gy - y)
        brg = math.atan2(rely, relx)

        # adaptive speed: scale v_max by forward free distance
        fc = self._min_sector(ranges, angle_min, angle_increment, -45, 45)
        v_lim = float(max(self.v_min, min(self.v_max, fc / self.speed_clear)))

        # DWA dynamic window: reachable velocities after one control step.
        look = 0.5
        vs = np.linspace(
            max(0.0, v_now - self.a_v * look),
            min(v_lim, v_now + self.a_v * look),
            18,
        )
        ws = np.linspace(
            max(-self.w_max, w_now - self.a_w * look),
            min(self.w_max, w_now + self.a_w * look),
            19,
        )

        best_cost = 1e9
        best_v, best_w = 0.0, 0.0

        if dist is None:
            return v_lim * max(0.0, math.cos(self._wrap(brg))), \
                float(np.clip(1.2 * self._wrap(brg), -self.w_max, self.w_max))

        for v in vs:
            for w in ws:
                cost = self._traj_cost(
                    v, w, half, res, dist, relx, rely, brg
                )
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w

        if best_cost == 1e9:
            # everything blocked (e.g. wedged against a box): rotating toward
            # the goal would grind into the obstacle, so pick the widest
            # goal-side gap instead and back out if still stuck.
            self._recover_mode = None
            gap = self._largest_gap(ranges, angle_min, angle_increment,
                                    goal_bearing=self._wrap(brg))
            if gap is not None:
                return 0.0, float(np.clip(self.rotate_gain * gap, -self.w_max, self.w_max))
            return 0.0, float(np.clip(1.2 * self._wrap(brg), -self.w_max, self.w_max))

        off_heading = abs(self._wrap(brg))
        goal_dist = math.hypot(relx, rely)
        fwd_free = self._min_sector(ranges, angle_min, angle_increment, -40, 40)
        # Is the corridor toward the *goal* itself open? The box may sit beside
        # the robot (not in the forward cone), so checking only straight ahead
        # misses the case where rotating toward the goal aims into the box.
        goal_free = self._cone_free(
            ranges, angle_min, angle_increment, brg, math.radians(45))
        # Only treat the goal as "blocked" when it is close - if the goal is far
        # away a box intruding into the goal cone is a local dodging problem the
        # trajectory cost handles, not a reason to abandon normal driving (which
        # caused the robot to spin hunting a gap around a far-away waypoint).
        blocked_to_goal = goal_free < 0.22 and 0.8 < goal_dist < 2.0
        if follow_path:
            # look-ahead goal is reachable by construction; never gate on it.
            blocked_to_goal = False

        # The goal is "reachable" again once the forward cone is open and the
        # goal direction is not blocked. Requiring near-perfect alignment to
        # exit recovery is a trap: recovery rotates toward a *gap*, not the
        # goal, so if the goal sits to the side the robot can never satisfy an
        # alignment exit and spins forever. Normal DWA steers toward the goal
        # from any heading, so the only thing that must hold to safely leave
        # recovery is an open forward cone and an unblocked goal line.
        goal_reachable = (fwd_free > 0.30
                          and goal_dist > self.goal_tolerance
                          and goal_dist < 12.0       # was 4.0: capped recovery-exit
                          # so the robot would spin trying to reach a distant
                          # open-lane waypoint; a roaming goal is always reachable
                          # once the forward cone is open.
                          and not blocked_to_goal)

        # ---- recovery entry / continuation ----
        if self._recover_mode is not None and not goal_reachable:
            tw = self._recover_twist(
                x, y, yaw, brg, goal_dist, ranges, angle_min, angle_increment,
                vs, ws, half, res, dist, best_v, best_w, best_cost)
            if now - self._last_dbg > self._dbg_period:
                self._last_dbg = now
                gap = self._largest_gap(ranges, angle_min, angle_increment,
                                        goal_bearing=self._wrap(brg))
                print(f"[dwa-dbg] RECOVERY mode={self._recover_mode} "
                      f"brg={brg:.2f} yaw={yaw:.2f} off={off_heading:.2f} "
                      f"goal_dist={goal_dist:.2f} fwd_free={fwd_free:.2f} "
                      f"goal_free={goal_free:.2f} blocked_goal={blocked_to_goal} "
                      f"gap={gap if gap is None else round(gap, 2)} "
                      f"-> v={tw[0]:.2f} w={tw[1]:.2f}", flush=True)
            return tw

        if self._recover_mode is not None and goal_reachable:
            # escaped the box: drop recovery and resume normal DWA.
            self._recover_mode = None
            self._stall_t = None
            self._stall_xy = None
            self._reverse_t = None
            self._recover_gap = None

        # Rotate toward the goal only when the straight path there is actually
        # open. If a box sits right in front of the goal direction, committing
        # to that rotation just grinds the robot into it; fall into recovery
        # instead so it goes around.
        if off_heading > self.rotate_threshold and not blocked_to_goal \
                and not follow_path:
            turn = float(np.clip(self.rotate_gain * self._wrap(brg), -self.w_max, self.w_max))
            return 0.0, turn

        stalled = blocked_to_goal or (
            best_v < 0.04 and goal_dist > self.goal_tolerance
        )
        if not stalled:
            self._stall_t = None
            self._stall_xy = None
            self._reverse_t = None
            if best_v < 0.12 and now - self._last_dbg > self._dbg_period:
                self._last_dbg = now
                print(f"[dwa-dbg] NORMAL slow v={best_v:.2f} w={best_w:.2f} "
                      f"brg={brg:.2f} yaw={yaw:.2f} off={off_heading:.2f} "
                      f"goal_dist={goal_dist:.2f} fwd_free={fwd_free:.2f} "
                      f"goal_free={goal_free:.2f} blocked_goal={blocked_to_goal}",
                      flush=True)
            return float(best_v), float(best_w)

        self._recover_mode = "align"
        self._recover_gap = None
        return self._recover_twist(
            x, y, yaw, brg, goal_dist, ranges, angle_min, angle_increment,
            vs, ws, half, res, dist, best_v, best_w, best_cost)

    # ---- recovery: align to the widest goal-side gap, then drive through it
    #      instead of oscillating on the goal bearing. ----
    def _recover_twist(self, x, y, yaw, brg, goal_dist, ranges, angle_min,
                       angle_increment, vs, ws, half, res, dist,
                       best_v, best_w, best_cost):
        c, s = math.cos(yaw), math.sin(yaw)
        now = time.monotonic()

        # Lock the gap at the start of the recovery run; re-evaluating every
        # tick lets two similar gaps (east/west of a box) flip the robot back
        # and forth forever.
        if self._recover_gap is None:
            self._recover_gap = self._largest_gap(
                ranges, angle_min, angle_increment,
                goal_bearing=self._wrap(brg))
        gap = self._recover_gap
        if gap is None:
            return 0.0, float(np.clip(1.2 * self._wrap(brg), -self.w_max, self.w_max))

        # ---- align: rotate in place toward the locked gap. Spinning counts as
        #      progress here, so no reverse timer runs during align. ----
        if self._recover_mode == "align":
            err = self._wrap(gap)
            if abs(err) > 0.12:
                return 0.0, float(np.clip(self.rotate_gain * err, -self.w_max, self.w_max))
            self._recover_mode = "drive"
            self._stall_t = now
            self._stall_xy = (x, y)

        # ---- drive: no-progress timer only now, after alignment is done. ----
        if self._stall_t is None:
            self._stall_t = now
            self._stall_xy = (x, y)
        moved = math.hypot(x - self._stall_xy[0], y - self._stall_xy[1])
        if now - self._stall_t > 2.5 and moved < 0.15:
            self._reverse_t = now
        if self._reverse_t is not None:
            if now - self._reverse_t < 0.8:
                rear = self._min_sector(ranges, angle_min, angle_increment,
                                        lo=-180, hi=180, abs_min=135)
                if rear > 0.3:
                    return -0.30, 0.0
            self._reverse_t = None
            self._stall_t = None
            self._stall_xy = None
            self._recover_gap = None
            self._recover_mode = "align"
            return 0.0, 0.0

        # run DWA with a virtual goal one meter along the chosen gap so the
        # cost keeps the robot inside the free lane.
        vgx = x + 1.0 * math.cos(yaw + gap)
        vgy = y + 1.0 * math.sin(yaw + gap)
        vrelx = c * (vgx - x) + s * (vgy - y)
        vrely = -s * (vgx - x) + c * (vgy - y)
        vbrg = math.atan2(vrely, vrelx)
        vbest_cost = 1e9
        gv, gw = best_v, best_w
        for v in vs:
            for w in ws:
                cost = self._traj_cost(v, w, half, res, dist, vrelx, vrely, vbrg)
                if cost is not None and cost < vbest_cost:
                    vbest_cost = cost
                    gv, gw = v, w
        if vbest_cost == 1e9:
            return 0.0, float(np.clip(self.rotate_gain * gap, -self.w_max, self.w_max))
        return float(gv), float(gw)

    # ------------------------------------------------------------ helpers
    def _largest_gap(self, ranges, angle_min, angle_increment, goal_bearing=0.0):
        """Return the center bearing (rad) of the widest free angular gap.

        Only gaps whose center lies within +-150 deg of the goal bearing are
        candidates, so recovery never aims at the wide open area behind the
        robot (the classic oscillation). Among candidates the widest wins,
        weighted toward the goal side."""
        r = np.asarray(ranges, dtype=float)
        free = np.isfinite(r) & (r > 0.6)
        n = len(r)
        if not free.any():
            return None
        angles = angle_min + np.arange(n) * angle_increment
        best_score = -1e9
        best_ang = None
        i = 0
        while i < n:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < n and free[j]:
                j += 1
            width = j - i
            center = angles[i + (j - i) // 2]
            err = abs(self._wrap(center - goal_bearing))
            if err > math.radians(150):
                i = j
                continue
            score = width * 2.0 - err * 2.0
            if score > best_score:
                best_score = score
                best_ang = center
            i = j
        if best_ang is None:
            return None
        return self._wrap(best_ang)

    def _cone_free(self, ranges, angle_min, angle_increment, center, width):
        """Minimum lidar range within +-width of a center bearing (wrap-safe)."""
        r = np.asarray(ranges, dtype=float)
        a = angle_min + np.arange(len(r)) * angle_increment
        err = np.abs(self._wrap(a - center))
        mask = err <= width
        r = r[mask]
        r = r[np.isfinite(r)]
        if not len(r):
            return 1e9
        return float(np.min(r))

    def _min_sector(self, ranges, angle_min, angle_increment, lo_deg=0, hi_deg=0,
                    lo=None, hi=None, abs_min=0.0):
        """Minimum lidar range within a bearing cone. If abs_min > 0 the cone
        covers |bearing| >= abs_min (e.g. the rear hemisphere for backups)."""
        r = np.asarray(ranges, dtype=float)
        a = angle_min + np.arange(len(r)) * angle_increment
        ad = a * 180.0 / math.pi
        if abs_min > 0.0:
            mask = np.abs(ad) >= abs_min
        elif lo is not None and hi is not None:
            mask = (ad >= lo) & (ad <= hi)
        else:
            mask = (ad >= lo_deg) & (ad <= hi_deg)
        r = r[mask]
        r = r[np.isfinite(r)]
        if not len(r):
            return 1e9
        return float(np.min(r))

    def _scan_points(self, ranges, angle_min, angle_increment):
        ranges = np.asarray(ranges, dtype=float)
        mask = np.isfinite(ranges) & (ranges > 0.05) & (ranges < 15.0)
        if not mask.any():
            return None
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        r = ranges[mask]
        a = angles[mask]
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)

    def _traj_cost(self, v, w, half, res, dist, relx, rely, brg=0.0):
        """Potential-field trajectory cost (lower is better).

        Sums obstacle repulsion 1/(d+eps) along a constant-curvature arc of
        duration self.horizon, plus a term rewarding progress toward the goal.
        Returns None if any path point is inside the robot footprint / out of
        the grid (i.e. the candidate must be rejected).
        """
        T = self.horizon
        if abs(w) < 1e-6:
            dth = 0.0
        else:
            dth = w * T

        n_pts = self.n_steps
        obs_cost = 0.0
        last_x, last_y = 0.0, 0.0
        n = int(2 * half / res)

        for k in range(1, n_pts + 1):
            frac = k / n_pts
            if abs(w) < 1e-6:
                px = v * T * frac
                py = 0.0
            else:
                r_arc = v / w
                th = dth * frac
                px = r_arc * math.sin(th)
                py = r_arc * (1.0 - math.cos(th))
            last_x, last_y = px, py
            if dist is None:
                continue
            ix = int((px + half) / res)
            iy = int((py + half) / res)
            if not (0 <= ix < n and 0 <= iy < n):
                return None  # left the local map; not a useful candidate
            dcell = dist[iy, ix]
            _r = getattr(self, "_tmp_radius", self.robot_radius)
            if dcell < _r - 0.02:
                return None
            _cl = _r + self.inflation
            if dcell < _cl:
                obs_cost += (_cl - dcell) / _cl

        ex, ey = last_x, last_y
        goal_dist = math.hypot(relx - ex, rely - ey)
        goal_term = self.goal_pot * goal_dist

        heading_err = self._wrap(brg - dth)
        goal_term += self.goal_pot * 0.6 * (1.0 - math.cos(heading_err))

        total = self.obs_w * obs_cost + goal_term
        # Strongly bias against stopping: cost penalty shrinks as v grows, so
        # the planner prefers a nonzero crawl over a dead stop. This keeps the
        # robot flowing through tight gaps instead of stopping to rotate.
        total += self.vel_w * (self.v_max - v)
        # 稳定代价: 手臂前伸(夹取姿态)时重心高且靠前, 转弯的向心加速度
        # v*|w| 和急转 |w| 是翻车主因 -> 惩罚大 v*w 轨迹, 让 DWA 在
        # 障碍间选择更平缓的走线(用户建议的稳定代价函数)。
        stab_w = float(os.getenv("DWA_STAB_W", "2.5"))
        total += stab_w * (v * abs(w))
        total += 0.3 * abs(w)
        if v < 0.05:
            total += float(os.getenv("DWA_STOP_PEN", "1.2"))
        return total

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi
