#!/usr/bin/env python3
"""Supermarket Sorting - Task Scheduler.

Orchestrates the full pipeline: receives the competition task JSON on
/supermarket_sorting/task, then for every target in order drives the robot to
pick the product and place it on the delivery table.

The scheduler reuses the proven, self-contained pick-place pipeline by launching
one client subprocess per target. The client (supermarket_sorting_client.py, or
the DWA-navigation variant) navigates to the shelf slot, grasps the visually
detected product via /kele/detections, navigates to the delivery table and
places it, then shuts itself down on DONE. The scheduler waits for that and
advances to the next target.

Components it relies on (all on the same ROS domain):
  * /supermarket_sorting/task        -> task JSON (transient-local)
  * /kele/detections                 -> perception (kele_detect.py) must be running
  * pick-place client (subprocess)   -> the arm+nav pipeline for one kind
  * the sim server                   -> /MMK2_mujoco_node

Run (inside the client container, e.g. domain 96):
    python3 /workspace/baseline/task_scheduler.py --max-targets 45

Environment:
  CLIENT_KIND  : which client to launch per target (default "dwa")
  RESET_BETWEEN: restart the scene between targets (default "1") - the sim
                 server resets the robot to the start tile on restart_scene.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# --- paths inside the client container (bind-mounted baseline) ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(REPO_ROOT, "examples", "supermarket_sorting")
NAV = os.path.join(REPO_ROOT, "navigation")

CLIENT_PLAIN = os.path.join(EXAMPLES, "supermarket_sorting_client.py")
CLIENT_DWA = os.path.join(NAV, "supermarket_sorting_client_dwa.py")


def ros_python_path():
    import sysconfig
    base = []
    for p in sys.path:
        if p and ("/opt/ros" in p):
            base.append(p)
    return ":".join(base)


class TaskScheduler(Node):
    def __init__(self, max_targets=None, reset_between=True, client_kind="dwa",
                 skip_grasp=False):
        super().__init__("task_scheduler")
        self.max_targets = max_targets
        self.reset_between = reset_between
        self.per_target_timeout = float(os.getenv("SCHED_PER_TARGET_TIMEOUT", "240"))
        self.client_kind = client_kind
        self.skip_grasp = skip_grasp

        self.task = None
        self.targets = []
        self.done = 0
        self.running_target = None
        self._proc = None
        self._started = False

        # transient-local: receive the task the server published at startup
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/supermarket_sorting/task",
                                 self.cb_task, qos)
        self._control_pub = self.create_publisher(String, "/supermarket_sorting/control", 5)
        self.get_logger().info(
            "task_scheduler up; waiting for /supermarket_sorting/task")

    def cb_task(self, msg):
        # 运行中(或已完成)忽略新任务: 场景重置会重新发布任务, 若重置运行
        # 状态会从订单1重新开始(调度器自己发的 reset 触发的重放)。
        if self._started and not getattr(self, "_finished", False):
            return
        if getattr(self, "_finished", False):
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"task JSON parse error: {exc}")
            return
        targets = data.get("targets", [])
        self.targets = [t for t in targets if isinstance(t, dict) and t.get("kind")]
        if self.max_targets:
            self.targets = self.targets[: self.max_targets]
        self.task = data
        self.done = 0
        self._started = True
        self.get_logger().info(
            f"TASK received: {len(self.targets)} targets -> "
            f"{[t['kind'] for t in self.targets]}")
        self._launch_next()

    # ---------------------------------------------------------- client launch
    def _client_cmd(self, kind):
        if self.client_kind == "dwa":
            client = CLIENT_DWA
        else:
            client = CLIENT_PLAIN
        cmd = [sys.executable, client, "--kind", kind]
        if self.skip_grasp:
            cmd.append("--skip-grasp")
        extra = os.getenv("CLIENT_ARGS", "").strip()
        if extra:
            cmd.extend(extra.split())
        return cmd

    def _launch_next(self):
        if self.done >= len(self.targets):
            self.get_logger().info(
                f"ALL {len(self.targets)} TARGETS COMPLETED")
            return
        target = self.targets[self.done]
        kind = target["kind"]
        self.running_target = target
        self.get_logger().info(
            f"[{self.done + 1}/{len(self.targets)}] picking kind={kind} "
            f"id={target.get('id')}")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{EXAMPLES}:{NAV}:{REPO_ROOT}:{ros_python_path()}"
        )
        env["ROS_DOMAIN_ID"] = env.get("ROS_DOMAIN_ID", "96")
        env["RMW_IMPLEMENTATION"] = env.get(
            "RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
        cmd = self._client_cmd(kind)
        self.get_logger().info(f"launching: {' '.join(cmd)}")
        # tee the client's output to a per-target log so the run can be watched
        log_path = f"/tmp/target_{self.done + 1:02d}_{kind}.log"
        self._log_fh = open(log_path, "w")
        self.get_logger().info(f"client log -> {log_path}")
        self._proc = subprocess.Popen(
            cmd, env=env,
            stdout=self._log_fh, stderr=subprocess.STDOUT,
        )
        self._proc_t0 = time.monotonic()

    # ------------------------------------------------------------- main loop
    def spin_once(self, timeout=0.1):
        rclpy.spin_once(self, timeout_sec=timeout)
        if not self._started or self._proc is None:
            return
        rc = self._proc.poll()
        if rc is None:
            # safety: never let one order block the whole run - kill a client
            # that exceeds its budget and move on (competition is 10 min total).
            if time.monotonic() - self._proc_t0 > self.per_target_timeout:
                self.get_logger().error(
                    f"target {self.done + 1} exceeded {self.per_target_timeout:.0f}s "
                    f"- killing client and continuing")
                self._proc.kill()
                rc = self._proc.wait(timeout=3.0)
        if rc is None:
            return
        # client finished (DONE -> rclpy.shutdown -> exit 0)
        self._proc = None
        if getattr(self, "_log_fh", None) is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self.done += 1
        self.get_logger().info(
            f"client exited rc={rc}; target {self.done}/{len(self.targets)} done")
        if rc != 0 and self.done < len(self.targets):
            # 客户端失败(卡死/翻车): 重置场景(服务器重载 ~90s), 下一单从起点跑
            self.get_logger().error(
                "client failed rc=%s; resetting scene before next target" % rc)
            try:
                import json as _j
                from std_msgs.msg import String as _S
                msg = _S(data=_j.dumps(
                    {"action": "restart_scene", "seed": 0,
                     "use_gs": True, "randomize": False}))
                self._control_pub.publish(msg)
            except Exception as _e:
                self.get_logger().error("reset publish failed: %s" % _e)
            time.sleep(95)
        if self.reset_between and self.done < len(self.targets):
            # brief pause to let the sim settle, then continue
            pass
        self._launch_next()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-targets", type=int, default=None,
                        help="process only the first N targets")
    parser.add_argument("--no-reset", action="store_true",
                        help="do not reset the scene between targets")
    parser.add_argument("--client", default="dwa",
                        choices=["dwa", "plain"],
                        help="pick-place client variant")
    parser.add_argument("--skip-grasp", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = TaskScheduler(
        max_targets=args.max_targets,
        reset_between=not args.no_reset,
        client_kind=args.client,
        skip_grasp=args.skip_grasp,
    )
    try:
        while rclpy.ok():
            node.spin_once(timeout=0.1)
            # 全部订单完成后退出进程: 常驻的调度器会在场景重置/重新发布任务时
            # 再次收到 task 并重复启动客户端, 与新一轮调度器抢底盘控制。
            if node.targets and node.done >= len(node.targets):
                node.get_logger().info("all targets done; scheduler exiting")
                node._finished = True
                break
    except KeyboardInterrupt:
        pass
    finally:
        if node._proc is not None:
            node._proc.terminate()
            try:
                node._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                node._proc.kill()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
