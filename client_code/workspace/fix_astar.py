path = "/workspace/baseline/navigation/dwa_navigator.py"
with open(path, "r") as f:
    code = f.read()

old = """        sub = self.astar.lookahead_target((self.odom[0], self.odom[1]), self.goal, lookahead=1.5)
        tx, ty = sub if sub else self.goal
        lin, ang = self.planner.plan(
            self.odom, (tx, ty),
            self.scan_ranges, self.scan_angle_min, self.scan_angle_incr,
            follow_path=True,
        )"""

new = """        gx, gy = self.goal
        sub = self.astar.lookahead_target((self.odom[0], self.odom[1]), self.goal, lookahead=1.5)
        if sub:
            dx, dy = sub[0] - self.odom[0], sub[1] - self.odom[1]
            a2g = math.atan2(dy, dx)
            off = abs(((a2g - self.odom[2]) + math.pi) % (2*math.pi) - math.pi)
            if off < 1.5:
                gx, gy = sub
        lin, ang = self.planner.plan(
            self.odom, (gx, gy),
            self.scan_ranges, self.scan_angle_min, self.scan_angle_incr,
        )"""

src = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("patched A* skip")
