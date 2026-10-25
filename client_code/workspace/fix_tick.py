path = "/workspace/baseline/navigation/dwa_navigator.py"
with open(path, "r") as f:
    code = f.read()

# Wrap the tick body with try/except to capture crashes
old = """    def tick(self):
        tw = Twist()
        if self.odom is None or self.goal is None:
            self.cmd_pub.publish(tw)
            return
        if self.scan_ranges is None:
            self.get_logger().warn("no scan yet; holding")
            self.cmd_pub.publish(tw)
            return"""

new = """    def tick(self):
        try:
            self._tick_inner()
        except Exception as e:
            import sys
            print(f"[tick-crash] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.cmd_pub.publish(Twist())

    def _tick_inner(self):
        tw = Twist()
        if self.odom is None or self.goal is None:
            self.cmd_pub.publish(tw)
            return
        if self.scan_ranges is None:
            self.get_logger().warn("no scan yet; holding")
            self.cmd_pub.publish(tw)
            return"""

src = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("tick crash handler added")
