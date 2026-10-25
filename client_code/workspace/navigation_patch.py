import re
path = "/workspace/baseline/navigation/dwa_navigator.py"
with open(path, "r") as f:
    code = f.read()

# Add A* import if not already present
if "from astar_planner import AstarPlanner" not in code:
    code = code.replace(
        "from dwa_planner import DWAPlanner",
        "from dwa_planner import DWAPlanner\nfrom astar_planner import AstarPlanner"
    )

# Insert astar init after self.planner = DWAPlanner(...)
old_init = "        self.planner = DWAPlanner(v_max=v_max, w_max=w_max, rotate_threshold=rot_th)\n"
new_init = (
    "        self.planner = DWAPlanner(v_max=v_max, w_max=w_max, rotate_threshold=rot_th)\n"
    "        self.astar = AstarPlanner(self.planner)\n"
)
if "self.astar = AstarPlanner" not in code:
    code = code.replace(old_init, new_init, 1)

# Replace the goal_tolerance block to use A* subgoals
old_goal_block = (
    "        dist = math.hypot(gx - cx, gy - cy)\n"
    "        if dist < self.planner.goal_tolerance:\n"
    "            self.stop()\n"
    "            self.log(\"[INFO] goal reached\")\n"
    "            self.goal = None\n"
    "            return\n"
    "        lin, ang = self.planner.plan("
)
new_goal_block = (
    "        dist = math.hypot(gx - cx, gy - cy)\n"
    "        if dist < self.planner.goal_tolerance:\n"
    "            self.stop()\n"
    "            self.log(\"[INFO] goal reached\")\n"
    "            self.goal = None\n"
    "            return\n"
    "        sub = self.astar.lookahead_target((cx, cy), (gx, gy), lookahead=1.5)\n"
    "        tx, ty = sub if sub else (gx, gy)\n"
    "        lin, ang = self.planner.plan("
)
code = code.replace(old_goal_block, new_goal_block, 1)

# Replace the final self.goal usage to pass tx,ty instead of gx,gy
code = code.replace(
    "            goal=(self.goal.x, self.goal.y),",
    "            goal=(tx, ty),",
    1
)

with open(path, "w") as f:
    f.write(code)
print("patched")
