path = "/workspace/baseline/navigation/dwa_navigator.py"
with open(path, "r") as f:
    src = f.read()

# Already has astar import and self.astar? If not, check
if "self.astar" not in src:
    print("ERROR: no self.astar found"); exit(1)

# Fix tick() to use A* subgoal
old = """        lin, ang = self.planner.plan(
            self.odom, self.goal,
            self.scan_ranges, self.scan_angle_min, self.scan_angle_incr,
        )"""
new = """        sub = self.astar.lookahead_target((self.odom[0], self.odom[1]), self.goal, lookahead=1.5)
        tx, ty = sub if sub else self.goal
        lin, ang = self.planner.plan(
            self.odom, (tx, ty),
            self.scan_ranges, self.scan_angle_min, self.scan_angle_incr,
            follow_path=True,
        )"""
src = src.replace(old, new, 1)

with open(path, "w") as f:
    f.write(src)
print("tick patched OK")
