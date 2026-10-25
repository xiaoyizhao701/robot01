#!/usr/bin/env python3
"""
kele perception node for the Supermarket Sorting task.

Mirrors the reference material_detection_client/yolo_detect.py architecture
but is specialised for the kele bottle and outputs poses in the WORLD frame
(the client consumes world-frame targets directly via arm_to()).

Pipeline
--------
  /head_camera/color/image_raw            (RGB,  bgr8 / rgb8)
  /head_camera/aligned_depth_to_color/... (depth, mono16 in mm)
  /head_camera/color/camera_info          (K)
  /joint_states + /odom                   (drive MMK2FK -> camera-in-world)
        |
        v  2-D detector backend (Blob / GT / YOLO)  -> bbox centre (u,v)
        v  pixel2cam: deproject (u,v,depth) with K  -> camera-frame point
        v  T_cam_world @ p_cam (MMK2FK headeye site) -> WORLD point
        |
        v  publish /kele/detections (vision_msgs/Detection3DArray, world frame)
           publish /kele/result_image (debug overlay)

The camera->world transform uses the repo's MMK2FK.get_head_camera_pose(),
fed with the live base pose (odom) + slide/head joints (joint_states).  The
'headeye' site already carries the OpenGL->OpenCV optical-frame flip, so the
deprojected point maps to world with NO extra axis swap (validated to
0.0 mm round-trip error).
"""

import os
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

from discoverse.robots.mmk2.mmk2_fk import MMK2FK

from backends import GtProjectionBackend, BlobBackend, YoloBackend

LAYOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "retail_competition_layout.json")
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "products.pt")
# The discoverse package resolves DISCOVERSE_ROOT_DIR to its pip install path in
# qy_run_client, so point MMK2FK at the actual repo MJCF explicitly. The MJCF uses
# a __REPO_ROOT__ placeholder that must be replaced with the repo task dir.
MJCF_TASK_DIR = "/workspace/supermarket_sorting_task/examples/supermarket_sorting"
MJCF_SRC = MJCF_TASK_DIR + "/mjcf/retail_competition.xml"
MJCF_PATH = "/tmp/retail_competition_fk.xml"
with open(MJCF_SRC) as _f:
    _mjcf_txt = _f.read().replace("__REPO_ROOT__", MJCF_TASK_DIR)
with open(MJCF_PATH, "w") as _f:
    _f.write(_mjcf_txt)
ARUCO_DICT = cv2.aruco.DICT_4X4_250


class KeleDetectNode(Node):
    def __init__(self, backend="blob", pub_res_img=True, device="auto",
                 ckpt=DEFAULT_CKPT, conf=0.45):
        super().__init__("kele_detect")
        self.bridge = CvBridge()
        self.pub_res_img = pub_res_img

        # camera intrinsics (from camera_info)
        self.K = None
        self._depth_msg = None

        # live robot state for the camera->world transform
        self.fk = MMK2FK(mjcf_path=MJCF_PATH)
        self.base_pos = None        # [x, y, z]
        self.base_quat = None       # [w, x, y, z]
        self.slide = 0.0
        self.head = [0.0, 0.0]

        # detector backend (pluggable)
        self.backend_name = backend
        if backend == "gt":
            self.detector = GtProjectionBackend(LAYOUT_JSON)
        elif backend == "yolo":
            self.detector = YoloBackend(ckpt, conf_thresh=conf, device=device)
        else:
            self.detector = BlobBackend()
        self.get_logger().info(f"kele_detect up; backend={backend}")
        # aruco_id -> 货位世界坐标(aruco 码固定绑货位,定位精确)
        try:
            self.aruco_world = {s["aruco_id"]: np.array(s["world_position"], float)
                                for s in json.load(open(LAYOUT_JSON))}
        except Exception:
            self.aruco_world = {}

        # Aruco slot detector: binds a kele detection to the nearest shelf-slot Aruco ID.
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._aruco_params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        else:
            self._aruco_detector = None

        # subscriptions
        self.create_subscription(CameraInfo, "/head_camera/color/camera_info",
                                 self.camera_info_cb, 10)
        self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10)
        self.create_subscription(Image, "/head_camera/color/image_raw",
                                 self.rgb_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10)

        # publishers
        self.det_pub = self.create_publisher(Detection3DArray, "/kele/detections", 10)
        self.img_pub = self.create_publisher(Image, "/kele/result_image", 5)

    # ---- state callbacks ----
    def camera_info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def depth_cb(self, msg: Image):
        self._depth_msg = msg

    def js_cb(self, msg: JointState):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    # ---- camera->world transform from live state ----
    def camera_world_tmat(self):
        """4x4 camera(optical)->world built from odom + slide/head via MMK2FK."""
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        self.fk.set_left_arm_joints([0.0] * 6)
        self.fk.set_right_arm_joints([0.0] * 6)
        pos, quat = self.fk.get_head_camera_pose()   # quat wxyz, world
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def pixel_to_cam(self, u, v, depth_m):
        """Deproject a pixel + metric depth to a camera-optical-frame point."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return np.array([x, y, depth_m])

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        """Median depth (m) over a patch, ignoring zero (invalid) pixels."""
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) * 1e-3 if len(valid) else 0.0

    def box_depth_m(self, depth_img, u, v, bw, bh):
        """Median depth (m) over the detection box (clamped), ignoring invalid."""
        h, w = depth_img.shape[:2]
        r = min(60, max(8, max(bw, bh) // 2))
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0.0
        # 商品近距离远少于背景:取较低分位数(前 40%)更贴近商品表面
        return float(np.percentile(valid, 40)) * 1e-3

    def _detect_aruco(self, rgb):
        """Return a list of (marker_id, cx, cy) for Aruco markers in the RGB frame."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if self._aruco_detector is not None:
            corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._aruco_params)
        centers = []
        if ids is not None:
            for c, marker_id in zip(corners, ids.flatten()):
                c = c.reshape(-1, 2)
                cx, cy = c.mean(axis=0)
                centers.append((int(marker_id), float(cx), float(cy)))
        return centers

    @staticmethod
    def _nearest_aruco(centers, u, v):
        """Return the Aruco ID whose centre is nearest to pixel (u, v), else None."""
        if not centers:
            return None
        best = min(centers, key=lambda t: (t[1] - u) ** 2 + (t[2] - v) ** 2)
        return best[0]

    # ---- main RGB callback ----
    def rgb_cb(self, msg: Image):
        if self.K is None or self._depth_msg is None:
            return
        T_cam_world = self.camera_world_tmat()
        if T_cam_world is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)  # mono16, mm
        aruco_centers = self._detect_aruco(rgb)

        dets = self.detector.detect(rgb, depth, self.K, T_cam_world)

        out = []
        vis = rgb.copy() if self.pub_res_img else rgb
        for d in dets:
            u, v = int(d["x"]), int(d["y"])
            if "gt_world_pos" in d:
                p_world = np.asarray(d["gt_world_pos"], dtype=float)
                # Visibility gate: the rendered depth at the projected pixel must
                # match the product's expected camera-frame depth. This rejects
                # occluded products (a pure projection would still report them).
                pc = np.linalg.inv(T_cam_world) @ np.append(p_world, 1.0)
                depth_m = self.patch_depth_m(depth, u, v)
                if depth_m <= 0.0 or abs(depth_m - pc[2]) > 0.60:
                    continue
            else:
                # 深度取 YOLO 框范围内的中值(框大部分覆盖商品),避免框中心单点
                # 取到背景深度导致位置偏移。
                bw, bh = int(d.get("w", 40)), int(d.get("h", 40))
                depth_m = self.box_depth_m(depth, u, v, bw, bh) if (bw >= 8 and bh >= 8) \
                    else self.patch_depth_m(depth, u, v)
                if depth_m <= 0.0:
                    continue
                p_cam = self.pixel_to_cam(u, v, depth_m)
                p_world = (T_cam_world @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

            aruco_id = self._nearest_aruco(aruco_centers, u, v)
            # YOLO 框中心+深度反投影在近距离/大框时容易取到背景深度而偏移;
            # 一旦匹配到货位 ArUco,直接用该货位的精确世界坐标(码固定绑货位)。
            if aruco_id is not None and aruco_id in self.aruco_world:
                p_world = self.aruco_world[aruco_id].copy()
            rec = {"class": d["class"], "conf": d.get("conf", 0.0), "world": p_world,
                   "aruco_id": aruco_id, "u": u, "v": v,
                   "w": int(d.get("w", 20)), "h": int(d.get("h", 20))}
            # coord-bridge validation logging (GT backend only)
            if "gt_world_pos" in d:
                err = np.linalg.norm(p_world - d["gt_world_pos"]) * 1e3
                rec["gt_err_mm"] = err
                self.get_logger().info(
                    f"[{d.get('body','?')}] world={np.round(p_world,3)} "
                    f"gt={np.round(d['gt_world_pos'],3)} err={err:.2f}mm")
            out.append(rec)

            if self.pub_res_img:
                w, h = int(d["w"]), int(d["h"])
                cv2.rectangle(vis, (u - w // 2, v - h // 2),
                              (u + w // 2, v + h // 2), (0, 255, 0), 2)
                label = f"{d['class']} aruco:{aruco_id}" if aruco_id is not None else d["class"]
                cv2.putText(vis, f"{label} ({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f})",
                            (u - 60, v - h // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        self.publish_detections(out, msg.header.stamp)
        if self.pub_res_img:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

    def publish_detections(self, recs, stamp):
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        for r in recs:
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(r["class"])
            hyp.hypothesis.score = float(r["conf"])
            hyp.pose.pose.position.x = float(r["world"][0])
            hyp.pose.pose.position.y = float(r["world"][1])
            hyp.pose.pose.position.z = float(r["world"][2])
            det.results.append(hyp)
            if r.get("u") is not None:
                det.bbox.center.position.x = float(r["u"])
                det.bbox.center.position.y = float(r["v"])
                det.bbox.size.x = float(r.get("w", 20))
                det.bbox.size.y = float(r.get("h", 20))
            if r.get("aruco_id") is not None:
                ahyp = ObjectHypothesisWithPose()
                ahyp.hypothesis.class_id = f"aruco:{r['aruco_id']}"
                ahyp.hypothesis.score = 1.0
                det.results.append(ahyp)
            msg.detections.append(det)
        self.det_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="kele perception node")
    parser.add_argument("--backend", default="blob",
                        choices=["blob", "gt", "yolo"],
                        help="2-D detector backend (default: blob)")
    parser.add_argument("--no-result-image", action="store_true",
                        help="disable /kele/result_image publishing")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="YOLO inference device (default: auto)")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="YOLO checkpoint path")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    args = parser.parse_args()

    rclpy.init()
    node = KeleDetectNode(backend=args.backend, pub_res_img=not args.no_result_image,
                          device=args.device, ckpt=args.ckpt, conf=args.conf)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
