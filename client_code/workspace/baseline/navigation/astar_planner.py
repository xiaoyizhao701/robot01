"""A* global path planner over the accumulated lidar occupancy map.

Plugs into the DWA navigation stack (`dwa_navigation.DwaNavigationMixin` +
`dwa_planner.DWAPlanner`). Instead of steering the local DWA controller
straight at a fixed route waypoint - which makes the robot wiggle side to
side past every box and stop-turn at the 90 deg corners - A* plans a global
path around the *known* obstacles and the local planner tracks a look-ahead
point along that path. The path is re-planned automatically as the lidar map
grows, so a box that is discovered mid-run simply re-routes the robot without
it ever having to stop.

Map semantics (shared with DWAPlanner): -1 unknown, 0 free, 100 occupied.
A* inflates occupied cells by the robot radius (+ a small clearance) and treats
everything else as traversable, charging an extra `unknown_penalty` cost for
cells not yet seen so the robot prefers known-open lanes while still finding a
way through unexplored space (the corridor ahead of the lidar reach).

Public API used by the mixin:

    astar.lookahead_target(pose, goal)
        -> (x, y) world point ``lookahead`` meters ahead along the global path
           from ``pose`` to ``goal``, or None when no plan is available yet
           (the caller then falls back to the raw goal).
"""
import heapq
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

_DIRS8 = ((-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)),
          (-1, 0, 1.0), (1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
          (0, 1, 1.0), (1, 1, math.sqrt(2.0)))


class AstarPlanner:
    """A* global planner fed from a DWAPlanner's world occupancy grid.

    The grid source must expose ``map``, ``map_origin``, ``map_res``, ``map_n``
    and a monotonic ``map_ver`` that increments on every map update.
    """

    def __init__(self, map_source, robot_radius=0.30, clearance=0.05,
                 unknown_penalty=3.0, replan_dist=0.25, lookahead=1.05,
                 plan_period=0.2, dbg=True):
        self.src = map_source
        self.robot_radius = robot_radius
        self.clearance = clearance
        self.unknown_penalty = unknown_penalty
        self.replan_dist = replan_dist
        self.lookahead = lookahead
        self.plan_period = plan_period
        self.dbg = dbg
        # keep paths near the middle of free aisles instead of hugging
        # obstacles / threading 1-cell-wide cracks: cells within
        # ``bound_dist`` of an obstacle cost ``bound_w`` extra per meter
        # of proximity.
        self.bound_dist = 0.7
        self.bound_w = 3.0

        self._blocked = None
        self._edt = None
        self._blocked_ver = None
        self._path = None
        self._goal = None
        self._plan_pose = None
        self._plan_ver = None
        self._last_t = 0.0
        self._dbg_last = 0.0

    # ------------------------------------------------------------- grid
    def _get_blocked(self):
        """Inflated obstacle grid (True = blocked), cached per map version."""
        if self._blocked is not None and self._blocked_ver == self.src.map_ver:
            return self._blocked
        occ = self.src.map == 100
        rad = (self.robot_radius + self.clearance) / self.src.map_res
        edt_cells = distance_transform_edt(~occ)
        blocked = edt_cells < rad
        self._blocked = blocked
        self._edt = edt_cells * self.src.map_res
        self._blocked_ver = self.src.map_ver
        return blocked

    def _visible(self, a, b, blocked):
        """True if the straight segment a->b stays clear of inflated cells."""
        res = self.src.map_res
        ox, oy = self.src.map_origin
        n = self.src.map_n
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(d / (0.5 * res)))
        for k in range(1, steps):
            t = k / steps
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            ix = int((x - ox) / res)
            iy = int((y - oy) / res)
            if 0 <= ix < n and 0 <= iy < n and blocked[iy, ix]:
                return False
        return True

    # ------------------------------------------------------------- search
    def _run(self, start, goal):
        """Plan a collision-free path from start to goal (world frame).

        The A* search runs on a grid coarsened by ``CF`` (2x) for speed; the
        returned path points are polished with the fine-grid clearance map so
        corner cuts are still checked at full resolution.
        """
        src = self.src
        if src.map_origin is None:
            return None
        ox, oy = src.map_origin
        res = src.map_res
        n = src.map_n
        blocked = self._get_blocked()

        # ---- coarse grid for the search itself ----
        cf = 2
        res_a = res * cf
        cn = n // cf
        b0 = blocked[:cn * cf, :cn * cf]
        cb = b0.reshape(cn, cf, cn, cf).any(axis=(1, 3))
        cb = cb.copy()  # start cell may be flipped free; keep cache intact
        edt_a = distance_transform_edt(~cb) * res_a
        mapg = src.map
        unk = (mapg[:cn * cf, :cn * cf]
               .reshape(cn, cf, cn, cf) < 0).any(axis=(1, 3))

        def cell(wx, wy):
            return (max(0, min(cn - 1, int((wx - ox) / res_a))),
                    max(0, min(cn - 1, int((wy - oy) / res_a))))

        sx, sy = cell(start[0], start[1])
        gx, gy = cell(goal[0], goal[1])
        if (sx, sy) == (gx, gy):
            return [(goal[0], goal[1])]

        # never refuse to plan because the *goal* cell is inflated: walk it
        # back toward the robot until a free cell is found.
        if cb[gy, gx]:
            gx, gy = self._back_off(gx, gy, sx, sy, cb)
        # the start cell is where the robot already stands; treat as free.
        cb[sy, sx] = False

        for margin in (1.5, 3.0, 1e9):
            path = self._search(sx, sy, gx, gy, cb, edt_a, unk,
                                ox, oy, res_a, cn, margin)
            if path is not None:
                path = self._simplify(path, blocked)
                path = self._relax(path, blocked)
                return path
        return None

    def _back_off(self, gx, gy, sx, sy, blocked):
        """Return the nearest free cell walking the goal back toward start."""
        best = (gx, gy)
        best_d = abs(gx - sx) + abs(gy - sy)
        n = blocked.shape[0]
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < n and 0 <= ny < n and not blocked[ny, nx]:
                    d = abs(nx - sx) + abs(ny - sy)
                    if d < best_d:
                        best_d = d
                        best = (nx, ny)
        return best

    def _search(self, sx, sy, gx, gy, blocked, edt, unk,
                ox, oy, res, n, margin):
        """A* on the (possibly coarsened) grid, cropping the expansion box."""
        if margin >= 1e8:
            x0 = y0 = 0
            x1 = y1 = n - 1
        else:
            m = int(margin / res)
            x0 = max(0, min(sx, gx) - m)
            x1 = min(n - 1, max(sx, gx) + m)
            y0 = max(0, min(sy, gy) - m)
            y1 = min(n - 1, max(sy, gy) + m)

        goal_idx = gy * n + gx
        g_score = np.full(n * n, 1e18, dtype=np.float64)
        came = np.full(n * n, -1, dtype=np.int32)
        g_score[sy * n + sx] = 0.0
        heap = [(self._heur(sx, sy, gx, gy), 0.0, sy * n + sx)]

        while heap:
            f, g, idx = heapq.heappop(heap)
            if idx == goal_idx:
                return self._reconstruct(idx, came, ox, oy, res, n)
            if g > g_score[idx] + 1e-9:
                continue
            cx = idx % n
            cy = idx // n
            for dx, dy, dcost in _DIRS8:
                nx, ny = cx + dx, cy + dy
                if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                    continue
                if blocked[ny, nx]:
                    continue
                mult = self.unknown_penalty if unk[ny, nx] else 1.0
                nd = g + dcost * mult
                prox = edt[ny, nx]
                if prox < self.bound_dist:
                    nd += self.bound_w * (self.bound_dist - prox)
                nidx = ny * n + nx
                if nd < g_score[nidx]:
                    g_score[nidx] = nd
                    came[nidx] = idx
                    heapq.heappush(heap,
                                   (nd + self._heur(nx, ny, gx, gy), nd, nidx))
        return None

    @staticmethod
    def _heur(x, y, gx, gy):
        return math.hypot(x - gx, y - gy)

    def _reconstruct(self, idx, came, ox, oy, res, n):
        pts = []
        while idx != -1:
            cx = idx % n
            cy = idx // n
            pts.append((ox + (cx + 0.5) * res, oy + (cy + 0.5) * res))
            idx = came[idx]
        pts.reverse()
        return pts

    # ------------------------------------------------------- path polish
    def _simplify(self, pts, blocked):
        """Drop redundant points while every shortcut stays collision-free."""
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._visible(pts[i], pts[j], blocked):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _relax(self, pts, blocked, iters=4):
        """Gently pull interior points toward their neighbors' midpoint while
        keeping every segment collision-free, rounding the path's corners."""
        if len(pts) < 3:
            return pts
        cur = list(pts)
        for _ in range(iters):
            nxt = cur[:]
            for i in range(1, len(cur) - 1):
                mx = 0.5 * (cur[i - 1][0] + cur[i + 1][0])
                my = 0.5 * (cur[i - 1][1] + cur[i + 1][1])
                cand = (cur[i][0] + 0.5 * (mx - cur[i][0]),
                        cur[i][1] + 0.5 * (my - cur[i][1]))
                if (self._visible(cur[i - 1], cand, blocked)
                        and self._visible(cand, cur[i + 1], blocked)):
                    nxt[i] = cand
            cur = nxt
        return cur

    # ------------------------------------------------------------ public
    def _plan(self, pose, goal):
        now = time.monotonic()
        goal_changed = self._goal != (goal[0], goal[1])
        moved = (self.replan_dist + 1.0 if self._plan_pose is None
                 else math.hypot(pose[0] - self._plan_pose[0],
                                 pose[1] - self._plan_pose[1]))
        if (self._path is not None and not goal_changed
                and moved < self.replan_dist
                and self._plan_ver == self.src.map_ver
                and now - self._last_t < self.plan_period):
            return self._path
        self._last_t = now
        self._plan_pose = (pose[0], pose[1])
        self._goal = (goal[0], goal[1])
        self._plan_ver = self.src.map_ver
        new_path = self._run(pose, goal)
        if new_path is not None:
            self._path = new_path
        return self._path if self._path is not None else new_path

    def lookahead_target(self, pose, goal, lookahead=None):
        """Return the world point ``lookahead`` m *ahead along* the planned
        path, or the goal itself when the whole path is closer than that.

        The path starts at the robot's cell, so walk forward accumulating arc
        length from the robot's actual pose (not path[0], which can be up to a
        cell away and make a tiny first segment shoot the interpolation back to
        the robot - the bug that made the look-ahead collapse onto the robot and
        stop it). The look-ahead is therefore always strictly ahead of the robot.
        """
        if lookahead is None:
            lookahead = self.lookahead
        path = self._plan(pose, goal)
        if not path:
            return None
        # Drop any leading path points at/behind the robot.
        prx, pry = pose[0], pose[1]
        i = 0
        while i < len(path) - 1:
            if math.hypot(path[i][0] - prx, path[i][1] - pry) > 1e-3:
                break
            i += 1
        if i >= len(path) - 1:
            return path[-1]
        # Arc length from the robot pose forward.
        acc = math.hypot(path[i][0] - prx, path[i][1] - pry)
        prev = (prx, pry)
        for p in path[i:]:
            seg = math.hypot(p[0] - prev[0], p[1] - prev[1])
            a0, a1 = acc, acc + seg
            acc = a1
            if a1 >= lookahead:
                if seg > 1e-6:
                    t = (lookahead - a0) / seg
                    return (prev[0] + t * (p[0] - prev[0]),
                            prev[1] + t * (p[1] - prev[1]))
                return p
            prev = p
        return path[-1]
