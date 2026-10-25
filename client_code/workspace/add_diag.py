path = "/workspace/baseline/navigation/dwa_navigator.py"
with open(path, "r") as f:
    code = f.read()

old = """        lin, ang = self.planner.plan(
            self.odom, (gx, gy),
            self.scan_ranges, self.scan_angle_min, self.scan_angle_incr,
        )"""

new = """        try:
            lin, ang = self.planner.plan(
                self.odom, (gx, gy),
                self.scan_ranges, self.scan_angle_min, self.scan_angle_incr,
            )
        except Exception as e:
            import sys
            print(f"[plan-crash] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            lin, ang = 0.0, 0.0"""

src = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("diag added")
