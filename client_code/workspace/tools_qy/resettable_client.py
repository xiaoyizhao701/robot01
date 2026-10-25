#!/usr/bin/env python3
"""Run the packaged baseline client with a complete ROS 2 reset service."""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import rclpy
from std_srvs.srv import Trigger


TASK_ROOT = "/workspace/supermarket_sorting_task"
CLIENT_DIR = os.path.join(TASK_ROOT, "examples", "supermarket_sorting")
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from supermarket_sorting_client import (  # noqa: E402
    GRIP_OPEN,
    INIT_ARM_L,
    INIT_ARM_R,
    NAV_SHELF,
    PickPlaceClient,
)


class ResettablePickPlaceClient(PickPlaceClient):
    """Baseline client whose controller state can be reset without a restart."""

    def __init__(self) -> None:
        super().__init__()
        self.client_reset_service = self.create_service(
            Trigger,
            "/supermarket_sorting/reset_client",
            self.reset_client_state,
        )
        self.client_restart_service = self.create_service(
            Trigger,
            "/supermarket_sorting/restart_baseline",
            self.restart_baseline,
        )
        self.get_logger().info("client reset service ready: /supermarket_sorting/reset_client")
        self.get_logger().info(
            "baseline restart service ready: /supermarket_sorting/restart_baseline"
        )

    def restart_baseline(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Exit PID 1 after replying so Docker restarts the complete baseline.

        This restarts both the detector and controller; it is intentionally a
        debug-wrapper feature and is not part of the submission client.
        """
        response.success = True
        response.message = "full baseline container restart requested"
        self.get_logger().warning(response.message)
        # This client is the shell's foreground process.  Exiting it lets the
        # PID-1 wrapper run its trap (which stops YOLO) and exit; Docker's
        # unless-stopped policy then recreates both processes from scratch.
        threading.Timer(0.35, lambda: os._exit(0)).start()
        return response

    def reset_client_state(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        # Command targets and their smoothed/published state.
        self.tc = np.zeros(19)
        self.tc[5:11] = INIT_ARM_L
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[18] = GRIP_OPEN
        self.action = self.tc.copy()
        self.tc_prev = self.tc.copy()
        self.joint_move_ratio = np.ones(19)

        # Discard feedback from the pre-reset simulation. Fresh callbacks will
        # repopulate these before the control tick is allowed to run again.
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None
        self.jvel = None

        # Navigation and manipulation state machine.
        self.phase = NAV_SHELF
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.sub_idx = 0
        self.sub_entered = False
        self.deploy_set = False
        self.place_sub = 0
        self.arm_target_set = False

        # Perception state. No target or samples from the previous run survive.
        self.OBJECT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.det_buf.clear()
        self.target_locked = False

        # Timers, logs and acceleration ramp.
        now = self.now()
        self.state_t0 = now
        self.last_wait_log = 0.0
        self.last_deploy_wait_log = 0.0
        self.last_log = 0.0
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0

        # Immediately publish a stopped base and the canonical initial joints.
        self.publish()
        response.success = True
        response.message = "client controller, navigation and perception state reset"
        self.get_logger().info(response.message)
        return response


def main() -> None:
    rclpy.init()
    node = ResettablePickPlaceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
