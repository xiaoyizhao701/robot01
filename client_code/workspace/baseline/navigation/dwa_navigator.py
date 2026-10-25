"""Two-phase DWANavigator with Nav2-style global navigation.

Phases
------
Phase 0  PURSUIT  - the robot follows the fixed ROUTE_TO_FINISH waypoints with
                    a rate-limited pure-pursuit drive. This covers the open,
                    box-free segment: from the start lane up the east side and
                    west *above* the corridor wall (world x=0.53 blocks y in
                    [-3.72, 1.70], so the route goes north first, then west at
                    y=2.30, then it may enter the top of the corridor). No
                    stop-turn and no DWA gating: the lane is known-clear.

Phase 1  NAV       - once a goal arrives whose point lies inside the obstacle
                    corridor, A* plans a global collision-free path from the
                    robot's current pose to the FINAL route point (the delivery
                    table) over the accumulated lidar occupancy map, and the
                    DWA local controller tracks a look-ahead point along that
                    path. The path is re-planned automatically as the map grows,
                    so a box discovered mid-run re-routes the robot smoothly.
                    Intermediate corridor waypoints are skipped - one continuous
                    plan handles the whole corridor, so the robot never
                    stop-turns or threads a 1-cell crack at each waypoint.

Nav2 role mapping used here
---------------------------
  occupancy costmap  : DWAPlanner's world-frame grid (built from lidar)
  global planner     : AstarPlanner (SmacPlanner-style edge-tolerant A*)
  local controller   : DWAPlanner  (DWA controller, tracks a path look-ahead)
  recovery           : rotate-to-widest-gap then backup (DWAPlanner recovery
                       + the navigator's jail-extract reverse watchdog)

A goal arriving inside the corridor latches ``_nav_active``; it stays latched
until the robot reaches the final route point, so the dashboard's per-leg goal
republishes (every 2 s) are treated as one continuous nav request instead of
restarting the A* plan at each intermediate waypoint.
"""
import math
import time
import sys

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from dwa_planner import DWAPlanner
from astar_planner import AstarPlanner

# ---- obstacle corridor bounds (from retail_competition.xml / supermarket_map.md)
OBSTACLE_ZONE = dict(x_min=-2.47, x_max=0.50, y_min=-3.72, y_max=1.70)

# ---- known-static walls seeded into the occupancy grid (world frame).
# The corridor east board blocks x=0.53 for y in [-3.72,1.70]; the perimeter
# walls bound the floor; the shelf row is at y=3.32. Seeding these means the
# A* planner inflates them and never routes through or hugs a wall it already
# knows - the root cause of the earlier grazing-and-spinning.
STATIC_BOXES = [
    (0.50, -3.72, 0.60, 1.70),     # corridor_right_board (east wall of corridor)
    (-2.55, -3.72, -2.47, 1.70),   # corridor west wall
    (-2.5, -3.78, 2.5, -3.70),     # arena south perimeter
    (-2.5, 3.68, 2.5, 3.78),       # arena north perimeter
    (-2.55, 1.65, -2.45, 3.78),    # west perimeter / shelf return
    (2.45, -3.72, 2.55, 3.78),     # east perimeter
]

# ---- fixed route to the delivery desk (world frame). Up the open east side to
# (1.52,2.30), across the top to (-1.52,2.30), then down through the corridor and
# south, ending just NE of the delivery table (-1.94,-3.41). The bottom of the
# corridor is open, so the final approach cuts west at the bottom to the table.
ROUTE_TO_FINISH = [
    (1.52, 2.30),
    (-1.52, 2.30),
    (0.20, -0.70),
    (0.20, -1.70),
    (-1.20, -2.80),
    (-2.00, -3.35),
]
ROUTE_FINAL = ROUTE_TO_FINISH[-1]   # just NE of the delivery table (-1.94,-3.41)


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DWANavigator(Node):
    def __init__(self):
        super().__init__("dwa_navigator")
        v_max = float(os.getenv("DWA_V_MAX", "1.30"))
        w_max = float(os.getenv("DWA_W_MAX", "1.10"))
        a_lat = float(os.getenv("DWA_A_LAT", "0.90"))
        rot_th = float(os.getenv("DWA_ROTATE_THRESHOLD", "0.70"))
        robot_radius = float(os.getenv("DWA_ROBOT_RADIUS", "0.36"))
        obs_w = float(os.getenv("DWA_OBS_W", "4.0"))
        speed_clear = float(os.getenv("DWA_SPEED_CLEAR", "0.9"))
        # pursuit-phase tuning (open lane drive)
        self.p_v = float(os.getenv("DWA_PURSUIT_V", "0.70"))
        self.p_w = float(os.getenv("DWA_PURSUIT_W", "0.85"))
        self.max_turn = float(os.getenv("DWA_MAX_TURN", "0.85"))
        self.a_lat = a_lat
        self.get_logger().info(
            f"limits: v_max={v_max} w_max={w_max} a_lat={a_lat} rot_th={rot_th} "
            f"robot_r={robot_radius} obs_w={obs_w} speed_clear={speed_clear} "
            f"pursuit_v={self.p_v} pursuit_w={self.p_w}"
        )
        self.planner = DWAPlanner(
            v_max=v_max,
            w_max=w_max,
            robot_radius=robot_radius,
            obs_w=obs_w,
            speed_clear=speed_clear,
            rotate_threshold=rot_th,
        )
        self.astar = AstarPlanner(self.planner, robot_radius=robot_radius, clearance=0.05)
        # Persist the known-static walls in the world occupancy grid so both the
        # A* global planner and the DWA local costmap avoid them from the start
        # (no repeated re-scanning or grazing-and-spinning at a wall).
        self.planner.seed_static(STATIC_BOXES)

        self.odom = None
        self.scan_msg = None
        self.goal = None                 # raw incoming nav goal (for mode pick)
        self._nav_active = False         # latched once a goal lands in the corridor
        self._nav_target = None          # (x, y) goal the controller steers toward
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.odom_cb, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan", self.scan_cb, 10)
        self.create_subscription(PoseStamped, "/nav/goal", self.goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.timer = self.create_timer(0.1, self.tick)

        # jail-extract reverse watchdog (nav/recovery escape)
        self._jail_a = None
        self._jail_xy = None
        self._jail_yaw = 0.0
        self._jail_d = None
        self._jail_reverse_until = 0.0
        self._jail_count = 0
        self._dbg_next = 0.0
        self._lin_s = 0.0
        self._ang_s = 0.0
        self.get_logger().info(
            "dwa_navigator up (two-phase: pursuit -> obstacle-corridor nav); "
            "publish a goal on /nav/goal"
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _in_zone(x: float, y: float) -> bool:
        z = OBSTACLE_ZONE
        return (z["x_min"] <= x <= z["x_max"]) and (z["y_min"] <= y <= z["y_max"])

    @staticmethod
    def _min_sector(ranges, angle_min, angle_increment, lo_deg, hi_deg):
        r = None
        n = len(ranges)
        for i in range(n):
            ad = math.degrees(angle_min + i * angle_increment)
            if not (lo_deg <= ad <= hi_deg):
                continue
            v = ranges[i]
            if math.isfinite(v) and (r is None or v < r):
                r = v
        return 1e9 if r is None else r

    @staticmethod
    def _widest_gap(ranges, angle_min, angle_increment):
        """Return the bearing (rad) to the centre of the widest free laser gap,
        weighted toward gaps that are open far ahead. Returns None if nothing is
        open. Used as a fallback target when A* can't find a path, so the robot
        drives toward open space instead of spinning in place."""
        n = len(ranges)
        free = [math.isfinite(ranges[i]) and ranges[i] > 0.5 for i in range(n)]
        if not any(free):
            return None
        best = None
        best_score = -1e9
        i = 0
        while i < n:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < n and free[j]:
                j += 1
            width = j - i
            # pick the widest run; among equal widths, the one that reaches
            # furthest (a gap that opens into the aisle, not a dead end).
            far = max(ranges[k] for k in range(i, j) if math.isfinite(ranges[k]))
            score = width * 2.0 + min(far, 5.0)
            if score > best_score:
                best_score = score
                best = angle_min + (i + (j - i) // 2) * angle_increment
            i = j
        return wrap_to_pi(math.atan2(math.sin(best), math.cos(best))) if best is not None else None

    def _pursuit(self, cx, cy, yaw, gx, gy):
        """Legacy pure-pursuit drive. Kept as a fallback; the controller now
        always uses the DWA planner (obstacle-aware with recovery) even in the
        open lane, because stray boxes outside the corridor used to make this
        naive heading-follow wedge the robot against a wall and spin."""
        brg = math.atan2(gy - cy, gx - cx)
        err = wrap_to_pi(brg - yaw)
        ang = float(max(-self.p_w, min(self.p_w, 2.0 * err)))
        lin = self.p_v * max(0.0, math.cos(err))
        fc = self._min_sector(
            self.scan_msg.ranges, self.scan_msg.angle_min,
            self.scan_msg.angle_increment, -45, 45)
        lin = min(lin, max(0.15, fc - 0.25))
        return max(0.0, lin), ang

    # ------------------------------------------------------------- callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = (p.x, p.y, yaw)

    def scan_cb(self, msg):
        self.scan_msg = msg

    def goal_cb(self, msg):
        gx, gy = msg.pose.position.x, msg.pose.position.y
        self.goal = (gx, gy)
        # Phase is chosen in tick() from BOTH the robot pose and the goal, not
        # from the goal alone. Where the robot actually is matters: if it is
        # inside the corridor but the goal is outside (e.g. a stray/goal from
        # the start lane), expanding only the goal would blind the controller
        # into driving straight through a box. Once the robot is inside the
        # corridor we stay in global-nav mode so it can't hurt itself.
        self._nav_target = (gx, gy)

    def _pick_phase(self, cx, cy):
        """Return True (= use NAV / avoid) when the robot is inside, or moving
        to a point inside, the obstacle corridor. Use the actual per-leg goal
        (not a far latched ROUTE_FINAL) so the A* target stays local to where
        the robot should be - planning to the far table dragged the robot past
        the corridor and wedged it against the west wall."""
        gx, gy = self._nav_target
        if self._in_zone(cx, cy) or self._in_zone(gx, gy):
            self._nav_active = True
        elif self._nav_active and not self._in_zone(cx, cy):
            self._nav_active = False
        return self._nav_active

    # ------------------------------------------------------------- control
    def tick(self):
        try:
            tw = Twist()
            if self.odom is None or self.goal is None or self.scan_msg is None:
                self.cmd_pub.publish(tw)
                return
            cx, cy, yaw = self.odom

            # nothing to do / target reached (use the controller target, not the
            # raw per-leg goal, so reaching the corridor final ends the run).
            if self._nav_target is None:
                self.cmd_pub.publish(tw)
                return

            # ---- Phase selection (robot pose + goal, both considered) ------
            self._pick_phase(cx, cy)

            # ---- Goal-overlap guard ----------------------------------------
            # If the goal (or NAV look-ahead) lands on/next to an obstacle, the
            # DWA would spin trying to "reach" an occupied point. Project it to
            # the nearest free cell before steering, so the robot arcs to a
            # reachable spot instead of whirling in place.
            sgx, sgy = self.planner.safe_goal(self._nav_target[0], self._nav_target[1])
            if (sgx, sgy) != self._nav_target:
                self.get_logger().info(
                    f"goal ({self._nav_target[0]:.2f},{self._nav_target[1]:.2f}) "
                    f"in obstacle -> steer to free cell ({sgx:.2f},{sgy:.2f})")
                self._nav_target = (sgx, sgy)
            tx, ty = self._nav_target
            dist = math.hypot(tx - cx, ty - cy)
            if dist < self.planner.goal_tolerance:
                if self._nav_active:
                    self.get_logger().info(f"reached delivery ({dist:.3f} m); stopping")
                    self._nav_active = False
                self.cmd_pub.publish(tw)
                return

            # ---- Plan ------------------------------------------------------
            # Always plan a global A* path to the goal and track a look-ahead on
            # it, not just inside the box corridor. The start region has shelf /
            # aisle structures too, so a naive straight pursuit kept driving the
            # robot into a narrow gap between them and freezing there. Routing
            # through the (accumulated-lidar) A* map in every phase makes the
            # robot swing around all structures instead of grinding into them.
            pose = (cx, cy, yaw)
            target = self.astar.lookahead_target(pose, self._nav_target, lookahead=1.5)
            if target is not None:
                tx, ty = target
                lin, ang = self.planner.plan(
                    pose, (tx, ty),
                    self.scan_msg.ranges, self.scan_msg.angle_min,
                    self.scan_msg.angle_increment,
                    follow_path=True,
                )
            else:
                # A* could not find a path (e.g. the robot is wedged in a pocket
                # the accumulated lidar map shows as boxed in, or it is too close
                # to an obstacle to plan). Pointing straight at the goal here is
                # what spun the robot in place. Instead drive toward the widest
                # free gap - the DWA will steer into it and the next tick re-plans
                # from the opened space, so the robot works its way out instead of
                # whirling.
                gap = self._widest_gap(
                    self.scan_msg.ranges, self.scan_msg.angle_min,
                    self.scan_msg.angle_increment)
                if gap is None:
                    lin, ang = 0.0, 0.0
                else:
                    err = wrap_to_pi(gap - yaw)
                    lin = self.p_v * max(0.0, math.cos(err))
                    ang = float(max(-self.max_turn, min(self.max_turn, 2.0 * err)))
                    fc = self._min_sector(
                        self.scan_msg.ranges, self.scan_msg.angle_min,
                        self.scan_msg.angle_increment, -45, 45)
                    lin = min(lin, max(0.15, fc - 0.25))

            # ---- Smooth + slow the motion ----------------------------------
            # Low-pass the twist so the robot doesn't jerk between ticks and
            # bleeds the big -1.2 rad/s spin commands; caps the last speed.
            sm = 0.30
            self._lin_s = self._lin_s + sm * (lin - self._lin_s)
            self._ang_s = self._ang_s + sm * (ang - self._ang_s)
            lin, ang = self._lin_s, self._ang_s
            # micro-deadband: stop dithering when the command is tiny.
            if abs(lin) < 0.02:
                lin = 0.0
            if abs(ang) < 0.02:
                ang = 0.0

            # ---- jail-extract watchdog (reverse out of a wedge) ------------
            # Detects "no net progress" rather than a pure stall: the robot can
            # be dithering (rotating back-and-forth, yaw changing) yet never get
            # closer to the target - the earlier check only caught pure-spin or
            # pure-forward stalls, so a robot spinning in place between two boxes
            # was never freed. Track distance-to-target over a window and bail
            # out with a reverse if it isn't shrinking.
            now = time.monotonic()
            if now < self._jail_reverse_until:
                rev = -0.35 if self._jail_count < 4 else -0.50
                tw.linear.x = float(rev)
                tw.angular.z = 0.0
                self.cmd_pub.publish(tw)
                return
            intent = lin > 0.22 or abs(ang) > 0.35
            if not intent:
                self._jail_a = None
                self._jail_xy = None
                self._jail_d = None
            else:
                if self._jail_a is None:
                    self._jail_a = now
                    self._jail_xy = (cx, cy)
                    self._jail_yaw = yaw
                    self._jail_d = dist
                elif now - self._jail_a > 3.0:
                    moved = math.hypot(cx - self._jail_xy[0], cy - self._jail_xy[1])
                    gained = self._jail_d - dist
                    turned = abs(wrap_to_pi(yaw - self._jail_yaw))
                    # stalled if: spinning w/o turning, moving w/o translating,
                    # OR simply not gaining on the goal after 3s of intent.
                    stalled = (
                        (abs(ang) > 0.35 and turned < 0.12)
                        or (lin > 0.10 and moved < 0.10)
                        or (gained < 0.05 and dist > 0.30)
                    )
                    if stalled:
                        self._jail_count += 1
                        self._jail_a = None
                        self._jail_xy = None
                        self._jail_d = None
                        dur = 0.8 if self._jail_count < 4 else 1.4
                        self._jail_reverse_until = now + dur
                        print(
                            f"[jail-extract #{self._jail_count}] lin={lin:.2f} ang={ang:.2f} "
                            f"moved={moved:.3f} dYaw={turned:.2f} gained={gained:.3f} dist={dist:.2f}",
                            file=sys.stderr, flush=True)

            # ---- debug + publish -------------------------------------------
            if abs(ang) > 1e-4:
                lin = min(lin, self.a_lat / max(abs(ang), 1e-3))
            # Cap the commanded turn to stop the endless fast-spin (the -1.2 rad/s
            # whirls the robot saw were the planner rotating in place to look for a
            # gap). Keep it well under w_max so the robot turns deliberately.
            if ang > self.max_turn:
                ang = self.max_turn
            elif ang < -self.max_turn:
                ang = -self.max_turn
            nnow = time.monotonic()
            if nnow >= self._dbg_next:
                self._dbg_next = nnow + 1.0
                print(
                    f"[nav-dbg] ph={'nav' if self._nav_active else 'pursuit'} "
                    f"pose=({cx:.2f},{cy:.2f}) yaw={yaw:.2f} goal=({self._nav_target[0]:.2f},"
                    f"{self._nav_target[1]:.2f}) tgt=({tx:.2f},{ty:.2f}) "
                    f"dist={dist:.2f} -> lin={lin:.2f} ang={ang:.2f}",
                    file=sys.stderr, flush=True)
            tw.linear.x = float(lin)
            tw.angular.z = float(ang)
            self.cmd_pub.publish(tw)
        except Exception as e:
            print(f"[tick-crash] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = DWANavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
