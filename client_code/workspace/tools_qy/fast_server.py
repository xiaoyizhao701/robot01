#!/usr/bin/env python3
"""Run the supermarket server with GPU EGL rendering and real-time pacing.

This is an external launcher: it changes no files in the packaged image.  It
adds selectable camera streams, wall-clock pacing, runtime telemetry, and safe
scene rebuild controls for the external dashboard.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import mujoco
from std_msgs.msg import String


# RTX 3090 is compute capability 8.6.  Avoid compiling gsplat helper kernels
# for every architecture visible in the installed CUDA toolkit.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


TASK_ROOT = Path("/workspace/supermarket_sorting_task")
SERVER_DIR = TASK_ROOT / "examples" / "supermarket_sorting"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from supermarket_sorting_server import (  # noqa: E402
    ScoredMMK2ROS2,
    build_config,
    spin_node,
)


def camera_ids(name: str, default: list[int]) -> list[int]:
    value = os.getenv(name)
    if value is None:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def install_incremental_gaussian_updates(renderer) -> None:
    """Update GS points only for MuJoCo bodies whose pose actually changed.

    The upstream renderer transforms every dynamic Gaussian on every camera
    frame.  In this scene that includes roughly 5.2 million points belonging
    to products resting on shelves.  Keeping per-body point ranges lets us
    update the moving robot (and any product being manipulated) without
    repeatedly transforming stationary products.
    """
    if not os.getenv("SUPERMARKET_INCREMENTAL_GS", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    import torch

    starts = renderer.gs_idx_start.tolist()
    ends = renderer.gs_idx_end.tolist()
    device = renderer.gaussians.device
    index_chunks = [
        torch.arange(start, end, dtype=torch.long, device=device)
        for start, end in zip(starts, ends)
    ]
    body_chunks = [
        torch.full((end - start,), index, dtype=torch.long, device=device)
        for index, (start, end) in enumerate(zip(starts, ends))
    ]
    last_pos: np.ndarray | None = None
    last_quat: np.ndarray | None = None

    @torch.no_grad()
    def update_gaussians_incremental(mj_data) -> None:
        nonlocal last_pos, last_quat
        pos = np.asarray(mj_data.xpos[renderer.gs_body_ids])
        quat = np.asarray(mj_data.xquat[renderer.gs_body_ids])
        if last_pos is None:
            changed = np.arange(len(starts))
        else:
            moved = np.max(np.abs(pos - last_pos), axis=1) > 1e-4
            # q and -q represent the same rotation.
            alignment = np.abs(np.sum(quat * last_quat, axis=1))
            rotated = alignment < 1.0 - 1e-6
            changed = np.flatnonzero(moved | rotated)
        last_pos, last_quat = pos.copy(), quat.copy()
        renderer.incremental_changed_bodies = int(changed.size)
        if changed.size == 0:
            return

        selected_indices = torch.cat([index_chunks[i] for i in changed])
        body_indices = torch.cat([body_chunks[i] for i in changed])
        pos_tensor = torch.as_tensor(pos, dtype=torch.float32, device=device)
        quat_tensor = torch.as_tensor(quat, dtype=torch.float32, device=device)
        point_pos = pos_tensor[body_indices]
        point_quat = quat_tensor[body_indices]
        xyz_original = renderer.gau_ori_xyz_all_cu[selected_indices]
        rot_original = renderer.gau_ori_rot_all_cu[selected_indices]

        quat_w = point_quat[:, :1]
        quat_xyz = point_quat[:, 1:]
        cross = 2.0 * torch.cross(quat_xyz, xyz_original, dim=-1)
        xyz = (
            xyz_original
            + quat_w * cross
            + torch.cross(quat_xyz, cross, dim=-1)
            + point_pos
        )

        w1, x1, y1, z1 = point_quat.unbind(-1)
        w2, x2, y2, z2 = rot_original.unbind(-1)
        rotation = torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )
        renderer.gaussians.xyz[selected_indices] = xyz
        renderer.gaussians.rot[selected_indices] = rotation

    renderer.update_gaussians = update_gaussians_incremental
    renderer.incremental_changed_bodies = len(starts)
    renderer.incremental_total_bodies = len(starts)
    renderer.incremental_total_points = int(renderer.gaussians.xyz.shape[0])
    print(
        "[fast-server] incremental GS updates enabled: %d bodies, %d points"
        % (len(starts), renderer.incremental_total_points),
        flush=True,
    )


def install_hybrid_gaussian_render(node) -> None:
    """Render sensors efficiently while preserving the native preview window.

    In headless/EGL mode the wrist streams use cheap MuJoCo rendering.  A
    visible GLFW window on this host is llvmpipe, so windowed mode instead
    batches head, wrist and preview cameras through CUDA 3DGS; GLFW only blits
    the finished preview pixels.
    """
    enabled = os.getenv("SUPERMARKET_GS_HEAD_ONLY", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return

    wrist_divisor = max(1, int(os.getenv("SUPERMARKET_WRIST_RENDER_DIVISOR", "2")))
    preview_divisor = max(1, int(os.getenv("SUPERMARKET_PREVIEW_RENDER_DIVISOR", "2")))
    render_frame = 0

    def hybrid_render(self) -> None:
        nonlocal render_frame
        self.render_cnt += 1
        width = int(self.config.render_set["width"])
        height = int(self.config.render_set["height"])
        self.update_renderer_window_size(width, height)
        windowed = not self.config.headless and self.window is not None
        wrist_due = render_frame % wrist_divisor == 0
        preview_due = windowed and render_frame % preview_divisor == 0

        gs_camera_ids = {
            camera_id
            for camera_id in (
                self.config.obs_rgb_cam_id + self.config.obs_depth_cam_id
            )
            if camera_id == 0
        }
        if windowed and wrist_due:
            gs_camera_ids.update(
                camera_id
                for camera_id in self.config.obs_rgb_cam_id
                if camera_id != 0
            )
        if preview_due:
            gs_camera_ids.add(self.cam_id)

        converted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if gs_camera_ids:
            self.gs_renderer.update_gaussians(self.mj_data)
            tensors = self.gs_renderer.render(
                self.mj_model,
                self.mj_data,
                sorted(gs_camera_ids),
                width,
                height,
                self.free_camera,
            )
            converted = self._convert_gs_render_results(tensors)
            for camera_id, (rgb, depth) in converted.items():
                if camera_id not in self.config.obs_rgb_cam_id:
                    continue
                self.img_rgb_obs_s[camera_id] = rgb
                if camera_id in self.config.obs_depth_cam_id:
                    self.img_depth_obs_s[camera_id] = depth

        if wrist_due and not windowed:
            previous_depth_mode = self.renderer._depth_rendering
            self.renderer.disable_depth_rendering()
            for camera_id in self.config.obs_rgb_cam_id:
                if camera_id != 0:
                    self.img_rgb_obs_s[camera_id] = self.getRgbImg(camera_id)
            self.renderer.enable_depth_rendering()
            for camera_id in self.config.obs_depth_cam_id:
                if camera_id != 0:
                    self.img_depth_obs_s[camera_id] = self.getDepthImg(camera_id)
            self.renderer._depth_rendering = previous_depth_mode

        # Preserve Discoverse's built-in interactive GLFW preview while the
        # sensor streams keep using the optimized hybrid render path.
        if windowed:
            import cv2
            import glfw
            import OpenGL.GL as gl

            if glfw.window_should_close(self.window):
                self.running = False
                return
            glfw.poll_events()
            display_result = converted.get(self.cam_id) if preview_due else None
            if display_result is not None:
                preview_rgb, preview_depth = display_result
                if self.renderer._depth_rendering:
                    preview = cv2.applyColorMap(
                        cv2.convertScaleAbs(
                            preview_depth,
                            alpha=255.0 / self.config.max_render_depth,
                        ),
                        cv2.COLORMAP_JET,
                    )
                    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                else:
                    preview = preview_rgb
                window_width, window_height = self.get_current_window_size()
                glfw.make_context_current(self.window)
                fb_width, fb_height = glfw.get_framebuffer_size(self.window)
                gl.glViewport(0, 0, fb_width, fb_height)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                preview = np.ascontiguousarray(preview[::-1])
                gl.glWindowPos2i(0, 0)
                gl.glPixelZoom(
                    window_width / max(1, preview.shape[1]),
                    window_height / max(1, preview.shape[0]),
                )
                gl.glDrawPixels(
                    preview.shape[1],
                    preview.shape[0],
                    gl.GL_RGB,
                    gl.GL_UNSIGNED_BYTE,
                    preview.tobytes(),
                )
                gl.glPixelZoom(1.0, 1.0)
                glfw.swap_buffers(self.window)
        render_frame += 1

    node.render = types.MethodType(hybrid_render, node)
    node.gs_hybrid_render = True
    node.gs_windowed_render = not node.config.headless
    node.wrist_render_divisor = wrist_divisor
    node.preview_render_divisor = preview_divisor
    print(
        "[fast-server] hybrid renderer enabled: head=3DGS, wrists=%s/%d, preview=%s/%d"
        % (
            "3DGS" if not node.config.headless else "MuJoCo EGL",
            wrist_divisor,
            "3DGS native window" if not node.config.headless else "off",
            preview_divisor,
        ),
        flush=True,
    )


class RuntimeControl:
    """Thread-safe commands sent by the visual dashboard."""

    def __init__(self, target_rtf: float) -> None:
        self._lock = threading.Lock()
        self._target_rtf = target_rtf
        self._restart: dict[str, object] | None = None
        self._manual_enabled = False
        self._manual_linear = 0.0
        self._manual_angular = 0.0
        self.status = "ready"

    def on_command(self, msg: String) -> None:
        try:
            command = json.loads(msg.data)
            action = command.get("action")
            with self._lock:
                if action == "set_rate":
                    value = float(command["target_rtf"])
                    if not 0.0 <= value <= 4.0:
                        raise ValueError("target_rtf must be between 0 and 4")
                    self._target_rtf = value
                    self.status = "rate updated"
                elif action == "restart_scene":
                    seed = int(command["seed"])
                    use_gs = command["use_gs"]
                    if not isinstance(use_gs, bool):
                        raise ValueError("use_gs must be boolean")
                    randomize = command.get(
                        "randomize",
                        os.getenv("SUPERMARKET_RANDOMIZE", "1") == "1",
                    )
                    if not isinstance(randomize, bool):
                        raise ValueError("randomize must be boolean")
                    self._restart = {
                        "seed": seed,
                        "use_gs": use_gs,
                        "randomize": randomize,
                    }
                    self.status = "scene restart requested"
                elif action == "manual_drive":
                    enabled = command.get("enabled")
                    if not isinstance(enabled, bool):
                        raise ValueError("enabled must be boolean")
                    linear = float(command.get("linear", 0.0))
                    angular = float(command.get("angular", 0.0))
                    if abs(linear) > 0.6 or abs(angular) > 1.2:
                        raise ValueError("manual drive command exceeds safety limit")
                    self._manual_enabled = enabled
                    self._manual_linear = linear
                    self._manual_angular = angular
                    self.status = "manual drive" if enabled else "baseline auto control"
                else:
                    raise ValueError("unknown action")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.status = "control error: %s" % exc
            print("[fast-server] %s" % self.status, flush=True)

    def target_rtf(self) -> float:
        with self._lock:
            return self._target_rtf

    def take_restart(self) -> dict[str, object] | None:
        with self._lock:
            restart, self._restart = self._restart, None
            return restart

    def manual_drive(self) -> tuple[bool, float, float]:
        with self._lock:
            return self._manual_enabled, self._manual_linear, self._manual_angular


def main() -> None:
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    config = build_config()
    config.obs_rgb_cam_id = camera_ids("SUPERMARKET_RGB_CAMS", [0, 1, 2])
    config.obs_depth_cam_id = camera_ids("SUPERMARKET_DEPTH_CAMS", [0])
    sensor_hz = float(os.getenv("SUPERMARKET_SENSOR_HZ", "24"))
    config.render_set["fps"] = sensor_hz
    target_rtf = float(os.getenv("SUPERMARKET_TARGET_RTF", "1.0"))

    gs_scene_only = (
        config.use_gaussian_renderer
        and os.getenv("SUPERMARKET_GS_SCENE_ONLY", "1").lower()
        in {"1", "true", "yes", "on"}
    )
    if gs_scene_only:
        scene_models = {"background", *config.obj_list}
        removed = set(config.gs_model_dict) - scene_models
        config.gs_model_dict = {
            name: path
            for name, path in config.gs_model_dict.items()
            if name in scene_models
        }
        print(
            "[fast-server] scene-only GS: removed %d robot link models; robot remains in MuJoCo views"
            % len(removed),
            flush=True,
        )

    node = ScoredMMK2ROS2(config)
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    integrator_name = os.getenv("SUPERMARKET_INTEGRATOR", "implicitfast").lower()
    try:
        node.mj_model.opt.integrator = integrators[integrator_name]
    except KeyError as exc:
        raise ValueError(
            "SUPERMARKET_INTEGRATOR must be one of %s" % sorted(integrators)
        ) from exc
    runtime_pub = node.create_publisher(String, "/supermarket_sorting/runtime", 5)
    control = RuntimeControl(target_rtf)
    node.create_subscription(
        String, "/supermarket_sorting/control", control.on_command, 5
    )
    if node.config.use_gaussian_renderer and hasattr(node, "gs_renderer"):
        install_incremental_gaussian_updates(node.gs_renderer)
        install_hybrid_gaussian_render(node)
    node.reset()
    spin_thread = threading.Thread(target=spin_node, args=(node,), daemon=True)
    spin_thread.start()
    publish_thread = threading.Thread(
        target=node.thread_pubros2topic, args=(sensor_hz,), daemon=True
    )
    publish_thread.start()

    run_sim_start = pace_sim_start = float(node.mj_data.time)
    run_wall_start = pace_wall_start = time.perf_counter()
    report_wall = run_wall_start
    report_sim = run_sim_start
    previous_sim = run_sim_start
    last_runtime_pub = 0.0
    rtf_samples: deque[tuple[float, float]] = deque(
        [(run_wall_start, run_sim_start)]
    )
    restart_config: dict[str, object] | None = None
    boot_id = "%d-%d" % (os.getpid(), time.time_ns())
    print(
        "[fast-server] EGL/headless=%s, RGB=%s, depth=%s, sensor=%.1f Hz, "
        "integrator=%s, timestep=%.4fs, target RTF=%.2f"
        % (
            config.headless,
            config.obs_rgb_cam_id,
            config.obs_depth_cam_id,
            sensor_hz,
            integrator_name,
            node.mj_model.opt.timestep,
            target_rtf,
        ),
        flush=True,
    )

    try:
        while rclpy.ok() and node.running:
            manual_enabled, manual_linear, manual_angular = control.manual_drive()
            if manual_enabled:
                node.tctr_base[0] = (
                    manual_linear - manual_angular * node.wheel_distance
                ) / node.wheel_radius
                node.tctr_base[1] = (
                    manual_linear + manual_angular * node.wheel_distance
                ) / node.wheel_radius
            node.step(node.target_control)

            # Headless Discoverse does not perform its normal GLFW sync sleep.
            # Pace against simulation time so controller timers and validation
            # still see a real-time run instead of an uncontrolled fast-forward.
            sim_now = float(node.mj_data.time)
            now = time.perf_counter()
            if sim_now < previous_sim:
                # reset_run rewinds MuJoCo time to zero.  Start a new pacing
                # epoch immediately so a reset cannot trigger a fast-forward
                # while simulation time catches the old epoch.
                run_sim_start = pace_sim_start = report_sim = sim_now
                run_wall_start = pace_wall_start = report_wall = now
                rtf_samples.clear()
                rtf_samples.append((now, sim_now))
                print("[fast-server] simulation clock reset; pacing epoch rebased", flush=True)
            previous_sim = sim_now

            requested_rtf = control.target_rtf()
            if requested_rtf != target_rtf:
                target_rtf = requested_rtf
                pace_sim_start, pace_wall_start = sim_now, now
                print("[fast-server] target RTF changed to %.2f" % target_rtf, flush=True)

            if target_rtf > 0:
                deadline = pace_wall_start + (sim_now - pace_sim_start) / target_rtf
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

            now = time.perf_counter()
            rtf_samples.append((now, sim_now))
            while len(rtf_samples) > 2 and now - rtf_samples[0][0] > 3.0:
                rtf_samples.popleft()
            sample_wall = now - rtf_samples[0][0]
            actual_rtf = (
                (sim_now - rtf_samples[0][1]) / sample_wall if sample_wall > 0 else 0.0
            )
            total_wall = now - run_wall_start
            total_rtf = (
                (sim_now - run_sim_start) / total_wall if total_wall > 0 else 0.0
            )

            if now - last_runtime_pub >= 0.5:
                runtime = {
                    "boot_id": boot_id,
                    "seed": int(os.getenv("SUPERMARKET_SEED", "0")),
                    "randomize": os.getenv("SUPERMARKET_RANDOMIZE", "1") == "1",
                    "use_gs": bool(config.use_gaussian_renderer),
                    "renderer": (
                        "3DGS sensors + native preview"
                        if getattr(node, "gs_windowed_render", False)
                        else
                        "3DGS scene + MuJoCo robot/wrists"
                        if getattr(node, "gs_hybrid_render", False) and gs_scene_only
                        else "3DGS head + MuJoCo wrists"
                        if getattr(node, "gs_hybrid_render", False)
                        else "3DGS + MuJoCo"
                        if config.use_gaussian_renderer
                        else "MuJoCo EGL"
                    ),
                    "gl_backend": os.getenv("MUJOCO_GL", "default"),
                    "headless": bool(config.headless),
                    "integrator": integrator_name,
                    "timestep": float(node.mj_model.opt.timestep),
                    "sensor_hz": sensor_hz,
                    "wrist_render_hz": sensor_hz
                    / int(getattr(node, "wrist_render_divisor", 1)),
                    "rgb_cams": config.obs_rgb_cam_id,
                    "depth_cams": config.obs_depth_cam_id,
                    "target_rtf": target_rtf,
                    "actual_rtf": actual_rtf,
                    "total_rtf": total_rtf,
                    "sim_time": sim_now - run_sim_start,
                    "wall_time": total_wall,
                    "status": control.status,
                    "gs_scene_only": gs_scene_only,
                    "manual_drive": manual_enabled,
                    "manual_linear": manual_linear,
                    "manual_angular": manual_angular,
                }
                if hasattr(node, "gs_renderer"):
                    runtime["gs_changed_bodies"] = int(
                        getattr(node.gs_renderer, "incremental_changed_bodies", 0)
                    )
                    runtime["gs_total_bodies"] = int(
                        getattr(node.gs_renderer, "incremental_total_bodies", 0)
                    )
                    runtime["gs_total_points"] = int(
                        getattr(node.gs_renderer, "incremental_total_points", 0)
                    )
                runtime_pub.publish(String(data=json.dumps(runtime, separators=(",", ":"))))
                last_runtime_pub = now

            if now - report_wall >= 5.0:
                interval_rtf = (sim_now - report_sim) / (now - report_wall)
                print(
                    "[fast-server] sim=%.2fs interval_RTF=%.3f total_RTF=%.3f"
                    % (sim_now, interval_rtf, total_rtf),
                    flush=True,
                )
                report_wall, report_sim = now, sim_now

            restart_config = control.take_restart()
            if restart_config is not None:
                print(
                    "[fast-server] restarting scene with seed=%s randomize=%s use_gs=%s"
                    % (
                        restart_config["seed"],
                        restart_config["randomize"],
                        restart_config["use_gs"],
                    ),
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        publish_thread.join(timeout=2.0)
        if getattr(node, "referee", None) is not None:
            try:
                node.referee.save_results()
            except Exception as exc:  # noqa: BLE001
                print("[fast-server] referee save_results failed: %s" % exc)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if restart_config is not None:
        os.environ["SUPERMARKET_SEED"] = str(restart_config["seed"])
        os.environ["SUPERMARKET_USE_GS"] = "1" if restart_config["use_gs"] else "0"
        os.environ["SUPERMARKET_RANDOMIZE"] = (
            "1" if restart_config["randomize"] else "0"
        )
        os.environ["SUPERMARKET_TARGET_RTF"] = str(target_rtf)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])


if __name__ == "__main__":
    main()
