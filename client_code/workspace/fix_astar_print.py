path = "/workspace/baseline/navigation/astar_planner.py"
with open(path, "r") as f:
    code = f.read()

old = """        if self.dbg and time.monotonic() - self._dbg_last > 1.0:
            self._dbg_last = time.monotonic()
            print(f"[astar-dbg] pose=({pose[0]:.2f},{pose[1]:.2f}) "
                  f"goal=({goal[0]:.2f},{goal[1]:.2f}) "
                  f"tgt=({target[0]:.2f},{target[1]:.2f}) n={len(path)}",
                  flush=True)"""

new = """        if self.dbg and time.monotonic() - self._dbg_last > 1.0:
            self._dbg_last = time.monotonic()
            try:
                print(f"[astar-dbg] pose=({pose[0]:.2f},{pose[1]:.2f}) "
                      f"goal=({goal[0]:.2f},{goal[1]:.2f}) "
                      f"tgt=({target[0]:.2f},{target[1]:.2f}) n={len(path)}",
                      flush=True)
            except Exception:
                pass"""

src = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("astar print fixed")
