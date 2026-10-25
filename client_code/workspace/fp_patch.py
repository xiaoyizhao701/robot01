path = "/workspace/baseline/navigation/dwa_navigator.py"
with open(path, "r") as f:
    code = f.read()

# Insert follow_path=True after v_now=self.cur_ang,
old = "            v_now=self.cur_lin, w_now=self.cur_ang,"
new = "            v_now=self.cur_lin, w_now=self.cur_ang, follow_path=True,"
code = code.replace(old, new, 1)

with open(path, "w") as f:
    f.write(code)
print("patched follow_path")
