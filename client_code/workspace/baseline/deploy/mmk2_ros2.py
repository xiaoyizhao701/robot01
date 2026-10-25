import threading
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, JointState, Imu, LaserScan

from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.utils import PIDarray, camera2k, get_site_tmat
import mujoco as _mj

class MMK2ROS2(MMK2Base, Node):
    target_control = np.zeros(19)
    def __init__(self, config: MMK2Cfg):
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)
        Node.__init__(self, 'MMK2_mujoco_node')

        self.pid_base_vel = PIDarray(
            kps=np.array([9.0, 9.0]),
            kis=np.array([1.0, 1.0]),
            kds=np.array([0.5, 0.5]),
            integrator_maxs=np.array([8.0, 8.0]),
        )

        self._raw_cmd = None
        self._raw_cmd_t = 0.0
        self._last_scan = None
        self.init_topic_publisher()
        self.init_topic_subscriber()

        if self.config.lidar_s2_sim:
            from mujoco_lidar.lidar_wrapper import MjLidarWrapper
            from mujoco_lidar.scan_gen import create_lidar_single_line

            self.lidar_frame_id = "laser"

            self.rays_theta, self.rays_phi = create_lidar_single_line(360, np.pi * 2.0)
            self.lidar_range_max = 12.0
            self.lidar_s2 = MjLidarWrapper(
                self.mj_model,
                site_name=self.lidar_frame_id,
                backend="cpu",
                cutoff_dist=self.lidar_range_max,
                args={"bodyexclude": self.mj_model.body("agv_link").id},
            )
            self.lidar_s2.trace_rays(
                self.mj_data, self.rays_theta, self.rays_phi)
            self.lidar_publish_frequency = 12.0
            self.next_lidar_publish_time = time.monotonic()

            self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            self.publish_static_transform(header_frame_id='base_link', child_frame_id=self.lidar_frame_id)

    def init_topic_publisher(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # rostopic joint state
        self.joint_state_puber = self.create_publisher(JointState, '/joint_states', 5)
        self.joint_state = JointState()
        self.joint_state.name = [
            "slide_joint", "head_yaw_joint", "head_pitch_joint",
            "left_arm_joint1" , "left_arm_joint2" , "left_arm_joint3" , "left_arm_joint4" , "left_arm_joint5" , "left_arm_joint6" , "left_arm_eef_gripper_joint" ,
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
        ]
        self.joint_state.position = self.sensor_qpos[2:].tolist()
        self.joint_state.velocity = self.sensor_qvel[2:].tolist()
        self.joint_state.effort = self.sensor_force[2:].tolist()

        # rostopic imu
        # # # self.imu_puber = self.create_publisher(Imu, '/imu', 5)
        # # self.imu_msg = Imu()
        # # self.imu_msg.header.frame_id = "mmk2_imu_link"

        # rostopic odometry
        self.odom_puber = self.create_publisher(Odometry, '/slamware_ros_sdk_server_node/odom', 5)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "/odom"
        # self.odom_msg.child_frame_id = "base_link"

        # image
        self.bridge = CvBridge()

        # lidar
        if self.config.lidar_s2_sim:
            self.lidar_s2_puber = self.create_publisher(LaserScan, '/slamware_ros_sdk_server_node/scan', 1)

        # internal (sim) clock for debug/timing: mj_data.time on /sim_clock
        self.sim_time_puber = self.create_publisher(Float64, '/sim_clock', 5)

        # image publisher, camera info publisher,  Initialize camera info messages
        if 0 in self.config.obs_rgb_cam_id:
            self.head_color_puber  = self.create_publisher(Image, '/head_camera/color/image_raw', 2)
            self.head_color_info_puber  = self.create_publisher(CameraInfo, '/head_camera/color/camera_info', 2)
            self.head_color_info = CameraInfo()
            self.head_color_info.width = self.config.render_set["width"]
            self.head_color_info.height = self.config.render_set["height"]
            self.head_color_info.k = camera2k(self.mj_model.cam_fovy[0] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 1 in self.config.obs_rgb_cam_id:
            self.left_color_puber  = self.create_publisher(Image, '/left_camera/color/image_raw', 2)
            self.left_color_info_puber  = self.create_publisher(CameraInfo, '/left_camera/color/camera_info', 2)
            self.left_color_info = CameraInfo()
            self.left_color_info.width = self.config.render_set["width"]
            self.left_color_info.height = self.config.render_set["height"]
            self.left_color_info.k = camera2k(self.mj_model.cam_fovy[1] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 2 in self.config.obs_rgb_cam_id:
            self.right_color_puber = self.create_publisher(Image, '/right_camera/color/image_raw', 2)
            self.right_color_info_puber = self.create_publisher(CameraInfo, '/right_camera/color/camera_info', 2)
            self.right_color_info = CameraInfo()
            self.right_color_info.width = self.config.render_set["width"]
            self.right_color_info.height = self.config.render_set["height"]
            self.right_color_info.k = camera2k(self.mj_model.cam_fovy[2] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 0 in self.config.obs_depth_cam_id:
            self.head_depth_puber  = self.create_publisher(Image, '/head_camera/aligned_depth_to_color/image_raw', 2)
            self.head_depth_info_puber  = self.create_publisher(CameraInfo, '/head_camera/aligned_depth_to_color/camera_info', 2)
            self.head_depth_info = CameraInfo()
            self.head_depth_info.width = self.config.render_set["width"]
            self.head_depth_info.height = self.config.render_set["height"]
            self.head_depth_info.k = camera2k(self.mj_model.cam_fovy[0] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()
 
        if (self.config.obs_rgb_cam_id is not None) or (self.config.obs_depth_cam_id is not None):            
            # Publish camera info periodically
            self.create_timer(1.0, self.publish_camera_info)

    def init_topic_subscriber(self):
        self.cmd_vel_suber = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 5)
        self.spine_cmd_suber = self.create_subscription(Float64MultiArray, '/spine_forward_position_controller/commands', self.cmd_spine_callback, 5)
        self.head_cmd_suber = self.create_subscription(Float64MultiArray, '/head_forward_position_controller/commands', self.cmd_head_callback, 5)
        self.left_arm_cmd_suber = self.create_subscription(Float64MultiArray, '/left_arm_forward_position_controller/commands', self.cmd_left_arm_callback, 5)
        self.right_arm_cmd_suber = self.create_subscription(Float64MultiArray, '/right_arm_forward_position_controller/commands', self.cmd_right_arm_callback, 5)
        # 抓取真值: 客户端告知目标商品编号, 服务端发布接触/提升/稳定状态
        self.grasp_target_suber = self.create_subscription(
            Float64, '/supermarket_sorting/grasp_target',
            self.grasp_target_callback, 5)
        self.grasp_status_puber = self.create_publisher(
            Float64MultiArray, '/supermarket_sorting/grasp_status', 5)
        self._grasp_target_body = None     # 目标商品的 body id
        self._grasp_target_init_z = None   # 目标商品初始高度(用于判定"被提升")

    def grasp_target_callback(self, msg: Float64):
        """收到目标商品编号(布局 product_NNN -> 运行时 item_run_*_NNN), 解析 body。"""
        nn = int(round(float(msg.data)))
        for b in range(self.mj_model.nbody):
            nm = _mj.mj_id2name(self.mj_model, _mj.mjtObj.mjOBJ_BODY, b)
            if nm and "item_run" in nm and nm.endswith(f"_{nn:02d}"):
                self._grasp_target_body = b
                self._grasp_target_init_z = None
                self.get_logger().info(f"[grasp] target body set: {nm} (id={b})")
                return
        self._grasp_target_body = None
        self.get_logger().warn(f"[grasp] no item_run body for NN={nn}")

    def _grasp_contact_pairs(self):
        """本 tick 处于接触的 (bodyid, bodyid) 对集合(双向)。"""
        pairs = set()
        gb = self.mj_model.geom_bodyid
        try:
            ncon = self.mj_data.ncon
        except AttributeError:
            return pairs
        for i in range(ncon):
            c = self.mj_data.contact[i]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            pairs.add((b1, b2))
            pairs.add((b2, b1))
        return pairs

    def _gripped(self, pairs, obj):
        """两指同时接触目标 = 夹持。"""
        fl = _mj.mj_name2id(self.mj_model, _mj.mjtObj.mjOBJ_BODY, "rgt_finger_left_link")
        fr = _mj.mj_name2id(self.mj_model, _mj.mjtObj.mjOBJ_BODY, "rgt_finger_right_link")
        touch = {fl, fr}
        left = (fl, obj) in pairs
        right = (fr, obj) in pairs
        return left, right, (left and right)

    def _publish_grasp_status(self, stamp):
        """发布目标商品接触/提升/稳定状态(客户端做最终成功判定)。"""
        msg = Float64MultiArray()
        data = [0.0] * 8     # 0..7: left, right, gripped, x, y, z, speed, tilt_deg
        if self._grasp_target_body is not None:
            obj = self._grasp_target_body
            pairs = self._grasp_contact_pairs()
            left, right, gripped = self._gripped(pairs, obj)
            x = float(self.mj_data.body(obj).xpos[0])
            y = float(self.mj_data.body(obj).xpos[1])
            z = float(self.mj_data.body(obj).xpos[2])
            if self._grasp_target_init_z is None:
                self._grasp_target_init_z = z
            # 速度: 物体 free joint 线速度
            jid = int(self.mj_model.body(obj).jntadr[0])
            dof = int(self.mj_model.jnt_dofadr[jid])
            speed = float(np.linalg.norm(self.mj_data.qvel[dof:dof + 3]))
            q = self.mj_data.body(obj).xquat
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
            data = [float(left), float(right), float(gripped),
                    x, y, z, speed, tilt]
        msg.data = data
        self.grasp_status_puber.publish(msg)

    def publish_camera_info(self):
        if 0 in self.config.obs_rgb_cam_id:
            self.head_color_info_puber.publish(self.head_color_info)
        if 1 in self.config.obs_rgb_cam_id:
            self.left_color_info_puber.publish(self.left_color_info)
        if 2 in self.config.obs_rgb_cam_id:
            self.right_color_info_puber.publish(self.right_color_info)
        if 0 in self.config.obs_depth_cam_id:
            self.head_depth_info_puber.publish(self.head_depth_info)

    def cmd_vel_callback(self, msg: Twist):
        self.tctr_base[0] = (msg.linear.x - msg.angular.z * self.wheel_distance) / self.wheel_radius
        self.tctr_base[1] = (msg.linear.x + msg.angular.z * self.wheel_distance) / self.wheel_radius
        self._raw_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self._raw_cmd_t = time.monotonic()

    def cmd_spine_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 1:
            self.tctr_slide[:] = msg.data[:]
        else:
            self.get_logger().error('Spine command length error')

    def cmd_head_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.tctr_head[:] = msg.data[:]
        else:
            self.get_logger().error("head command length error")

    def cmd_left_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_left_arm[:] = msg.data[:6]
            self.tctr_lft_gripper[:] = msg.data[6:]
        else:
            self.get_logger().error("left arm command length error")

    def cmd_right_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_right_arm[:] = msg.data[:6]
            self.tctr_rgt_gripper[:] = msg.data[6:]
        else:
            self.get_logger().error("right arm command length error")

    # 关节名 -> qpos adr
    def _arm_joint_adrs(self):
        adrs = []
        for nm in ("rgt_arm_joint1", "rgt_arm_joint2", "rgt_arm_joint3",
                   "rgt_arm_joint4", "rgt_arm_joint5", "rgt_arm_joint6"):
            jid = _mj.mj_name2id(self.mj_model, _mj.mjtObj.mjOBJ_JOINT, nm)
            adrs.append(self.mj_model.jnt_qposadr[jid] if jid >= 0 else -1)
        return adrs

    def filter_right_arm_collision(self):
        """主循环内调用: 评估 tctr_right_arm 姿态是否自碰/撞货架。

        直接在 mj_data 上设置关节 + mj_forward(纯运动学, 不步进), 检查接触后恢复。
        若碰撞, 把目标恢复为上一次安全姿态(防止顶死)。
        """
        if not hasattr(self, "_arm_jadr"):
            self._arm_jadr = self._arm_joint_adrs()
            self._last_safe_arm = self.tctr_right_arm.copy()
            self._collision_log_t = 0.0
        adr = self._arm_jadr
        if any(a < 0 for a in adr):
            return
        saved = self.mj_data.qpos[adr].copy()
        self.mj_data.qpos[adr] = self.tctr_right_arm
        try:
            _mj.mj_forward(self.mj_model, self.mj_data)
            bad = self._arm_has_bad_contact()
        except Exception:
            bad = False
        self.mj_data.qpos[adr] = saved
        _mj.mj_forward(self.mj_model, self.mj_data)
        if bad:
            self.tctr_right_arm[:] = self._last_safe_arm
            now = float(self.mj_data.time)
            if now - self._collision_log_t > 1.0:
                self._collision_log_t = now
                self.get_logger().warn(
                    f"[arm-collision] commanded arm pose collides; holding last safe pose")
        else:
            self._last_safe_arm = self.tctr_right_arm.copy()

    def _arm_has_bad_contact(self):
        """机械臂与 货架/商品/自身 的接触检查。mj_forward 不填 contact, 因此用
        mjData 的几何位置做简单 AABB/包围球距离判定(仅用于安全过滤)。"""
        gb = self.mj_model.geom_bodyid
        # 收集机械臂 geom 的 world 包围球
        arm_geoms = []
        for g in range(self.mj_model.ngeom):
            nm = _mj.mj_id2name(self.mj_model, _mj.mjtObj.mjOBJ_GEOM, g)
            if nm and ("rgt_arm" in nm or "rgt_finger" in nm) and "collision" in nm:
                arm_geoms.append((g, self.mj_model.geom_rbound[g]))
        if not arm_geoms:
            return False
        # 与货架/商品/墙 geom 的球-球距离检查(粗筛)
        for g in range(self.mj_model.ngeom):
            nm = _mj.mj_id2name(self.mj_model, _mj.mjtObj.mjOBJ_GEOM, g) or ""
            if not any(k in nm for k in ("shelf", "item_run", "table", "wall", "boundary")):
                continue
            r2 = self.mj_model.geom_rbound[g]
            p2 = self.mj_data.geom_xpos[g]
            for g1, r1 in arm_geoms:
                p1 = self.mj_data.geom_xpos[g1]
                d = np.linalg.norm(p1 - p2)
                if d < (r1 + r2) * 0.9:
                    return True
        return False

    def resetState(self):
        super().resetState()
        self._pin_qpos = None
        self.pid_base_vel.reset()
        self.target_control[:] = self.init_joint_ctrl[:]
        if self.config.lidar_s2_sim:
            self.next_lidar_publish_time = time.monotonic()

    def _min_range_cone(self, half_deg, rear=False):
        """Minimum lidar range inside the forward cone (±half_deg) or the rear
        cone (|bearing| >= 180-half_deg). Used to clamp the velocity drive so it
        never bulldozes into a close obstacle."""
        d = getattr(self, "_last_scan", None)
        th = getattr(self, "_scan_theta", None)
        if d is None or th is None:
            return 1e9
        best = 1e9
        lim = math.radians(half_deg)
        for i in range(len(d)):
            a = float(th[i])
            if a > math.pi:
                a -= 2.0 * math.pi
            hit = (abs(a) >= lim) if rear else (abs(a) <= lim)
            if hit:
                r = float(d[i])
                if r < best:
                    best = r
        return best

    def updateControl(self, action):
        # Direct base-velocity control + forward-obstacle clamp. The velocity
        # drive keeps the robot from freezing; the clamp stops it from driving
        # into a near obstacle (which the kinematic base would push through and
        # then be flipped by the contact impulse).
        self.mj_data.ctrl[0] = 0.0
        self.mj_data.ctrl[1] = 0.0
        fresh = (getattr(self, "_raw_cmd", None) is not None
                 and time.monotonic() - getattr(self, "_raw_cmd_t", 0.0) < 0.3)
        vx = vy = wz = 0.0
        if fresh:
            vx, vy, wz = self._raw_cmd
            if vx > 0.0 and self._min_range_cone(35.0) < 0.15:
                vx = 0.0
            elif vx < 0.0 and self._min_range_cone(180.0, rear=True) < 0.30:
                vx = 0.0
        if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
            # Parking brake: with no commanded motion, hold the base pose
            # exactly. Zeroing qvel alone is not enough -- arm reaction forces
            # still shift the base position within each physics step, which
            # accumulates into a slow drift that ruins the grasp alignment.
            if getattr(self, "_pin_qpos", None) is None:
                self._pin_qpos = self.mj_data.qpos[0:7].copy()
            else:
                self.mj_data.qpos[0:7] = self._pin_qpos
            self.mj_data.qvel[0:6] = 0.0
        else:
            self._pin_qpos = None
            q = self.mj_data.qpos[3:7]
            w = q[0]; x = q[1]; y = q[2]; z = q[3]
            yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
            c, s = math.cos(yaw), math.sin(yaw)
            self.mj_data.qvel[0] = vx * c - vy * s
            self.mj_data.qvel[1] = vx * s + vy * c
            self.mj_data.qvel[2] = 0.0
            self.mj_data.qvel[3] = 0.0
            self.mj_data.qvel[4] = 0.0
            self.mj_data.qvel[5] = wz
        # keep the arms/head/grippers on their position controllers
        self.mj_data.ctrl[2:self.njctrl] = np.clip(action[2:self.njctrl], self.mj_model.actuator_ctrlrange[2:self.njctrl,0], self.mj_model.actuator_ctrlrange[2:self.njctrl,1])

    def publish_static_transform(self, header_frame_id, child_frame_id):
        stfs_msg = TransformStamped()
        stfs_msg.header.stamp = self.get_clock().now().to_msg()
        stfs_msg.header.frame_id = header_frame_id
        stfs_msg.child_frame_id = child_frame_id

        tmat_base = get_site_tmat(self.mj_data, header_frame_id)
        tmat_child = get_site_tmat(self.mj_data, child_frame_id)
        tmat_trans = np.linalg.inv(tmat_base) @ tmat_child
        
        stfs_msg.transform.translation.x = tmat_trans[0, 3]
        stfs_msg.transform.translation.y = tmat_trans[1, 3]
        stfs_msg.transform.translation.z = tmat_trans[2, 3]

        quat = Rotation.from_matrix(tmat_trans[:3, :3]).as_quat()
        stfs_msg.transform.rotation.x = quat[0]
        stfs_msg.transform.rotation.y = quat[1]
        stfs_msg.transform.rotation.z = quat[2]
        stfs_msg.transform.rotation.w = quat[3]

        self.static_broadcaster.sendTransform(stfs_msg)

    def post_physics_step(self):
        # 主循环内: 机械臂指令碰撞过滤(自碰/撞货架则保持上一安全位)
        try:
            self.filter_right_arm_collision()
        except Exception:
            pass
        if not self.config.lidar_s2_sim:
            return

        now = time.monotonic()
        if now < self.next_lidar_publish_time:
            return

        self.publish_lidar_scan(self.lidar_publish_frequency)
        period = 1.0 / self.lidar_publish_frequency
        self.next_lidar_publish_time += period
        if self.next_lidar_publish_time <= now:
            self.next_lidar_publish_time = now + period

    def publish_lidar_scan(self, freq=12):
        if not self.config.lidar_s2_sim:
            return

        dists = self.lidar_s2.trace_rays(
            self.mj_data, self.rays_theta, self.rays_phi)
        dists = np.asarray(dists, dtype=np.float32)
        dists[(dists < 0.02) | (dists > self.lidar_range_max)] = np.inf
        self._last_scan = dists
        self._scan_theta = self.rays_theta

        scan_msg = LaserScan()
        scan_msg.header.frame_id = self.lidar_frame_id
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.angle_min = float(self.rays_theta[0])
        scan_msg.angle_max = float(self.rays_theta[-1])
        scan_msg.angle_increment = float(self.rays_theta[1] - self.rays_theta[0])
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / float(freq)
        scan_msg.range_min = 0.02
        scan_msg.range_max = self.lidar_range_max
        scan_msg.ranges = dists.tolist()
        scan_msg.intensities = []

        self.lidar_s2_puber.publish(scan_msg)

    def thread_pubros2topic(self, freq=30):
        rate = self.create_rate(freq)
        while rclpy.ok() and self.running:
            time_stamp = self.get_clock().now().to_msg()

            self.joint_state.header.stamp = time_stamp
            self.joint_state.position = self.sensor_qpos[2:].tolist()
            self.joint_state.velocity = self.sensor_qvel[2:].tolist()
            self.joint_state.effort = self.sensor_force[2:].tolist()
            self.joint_state_puber.publish(self.joint_state)

            self.odom_msg.header.stamp = time_stamp
            self.odom_msg.pose.pose.position.x = self.sensor_base_position[0]
            self.odom_msg.pose.pose.position.y = self.sensor_base_position[1]
            self.odom_msg.pose.pose.position.z = self.sensor_base_position[2]
            self.odom_msg.pose.pose.orientation.w = self.sensor_base_orientation[0]
            self.odom_msg.pose.pose.orientation.x = self.sensor_base_orientation[1]
            self.odom_msg.pose.pose.orientation.y = self.sensor_base_orientation[2]
            self.odom_msg.pose.pose.orientation.z = self.sensor_base_orientation[3]
            self.odom_msg.twist.twist.linear.x = self.sensor_base_linear_vel[0]
            self.odom_msg.twist.twist.linear.y = self.sensor_base_linear_vel[1]
            self.odom_msg.twist.twist.linear.z = self.sensor_base_linear_vel[2]
            self.odom_msg.twist.twist.angular.x = self.sensor_base_gyro[0]
            self.odom_msg.twist.twist.angular.y = self.sensor_base_gyro[1]
            self.odom_msg.twist.twist.angular.z = self.sensor_base_gyro[2]
            self.odom_puber.publish(self.odom_msg)
            self.sim_time_puber.publish(Float64(data=float(self.mj_data.time)))
            self._publish_grasp_status(time_stamp)
            
            trans_msg = TransformStamped()
            trans_msg.header.stamp = time_stamp
            trans_msg.header.frame_id = "odom"
            trans_msg.child_frame_id = "base_link"
            trans_msg.transform.translation.x = self.sensor_base_position[0]
            trans_msg.transform.translation.y = self.sensor_base_position[1]
            trans_msg.transform.translation.z = self.sensor_base_position[2]
            trans_msg.transform.rotation.w = self.sensor_base_orientation[0]
            trans_msg.transform.rotation.x = self.sensor_base_orientation[1]
            trans_msg.transform.rotation.y = self.sensor_base_orientation[2]
            trans_msg.transform.rotation.z = self.sensor_base_orientation[3]            
            self.tf_broadcaster.sendTransform(trans_msg)

            # self.imu_msg.header.stamp = time_stamp
            # self.imu_msg.orientation.w = self.sensor_base_orientation[0]
            # self.imu_msg.orientation.x = self.sensor_base_orientation[1]
            # self.imu_msg.orientation.y = self.sensor_base_orientation[2]
            # self.imu_msg.orientation.z = self.sensor_base_orientation[3]
            # self.imu_msg.angular_velocity.x = self.sensor_base_gyro[0]
            # self.imu_msg.angular_velocity.y = self.sensor_base_gyro[1]
            # self.imu_msg.angular_velocity.z = self.sensor_base_gyro[2]
            # self.imu_msg.linear_acceleration.x = self.sensor_base_acc[0]
            # self.imu_msg.linear_acceleration.y = self.sensor_base_acc[1]
            # self.imu_msg.linear_acceleration.z = self.sensor_base_acc[2]
            # # self.imu_puber.publish(self.imu_msg)

            if 0 in self.config.obs_rgb_cam_id:
                head_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][0], encoding="rgb8")
                head_color_img_msg.header.stamp = time_stamp
                head_color_img_msg.header.frame_id = "head_camera"
                self.head_color_puber.publish(head_color_img_msg)

            if 1 in self.config.obs_rgb_cam_id:
                left_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][1], encoding="rgb8")
                left_color_img_msg.header.stamp = time_stamp
                left_color_img_msg.header.frame_id = "left_camera"
                self.left_color_puber.publish(left_color_img_msg)

            if 2 in self.config.obs_rgb_cam_id:
                right_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][2], encoding="rgb8")
                right_color_img_msg.header.stamp = time_stamp
                right_color_img_msg.header.frame_id = "right_camera"
                self.right_color_puber.publish(right_color_img_msg)

            if 0 in self.config.obs_depth_cam_id:
                head_depth_img = np.array(np.clip(self.obs["depth"][0]*1e3, 0, 65535), dtype=np.uint16)
                head_depth_img_msg = self.bridge.cv2_to_imgmsg(head_depth_img, encoding="mono16")
                head_depth_img_msg.header.stamp = time_stamp
                head_depth_img_msg.header.frame_id = "head_camera"
                self.head_depth_puber.publish(head_depth_img_msg)

            rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = [0,1,2]
    cfg.obs_depth_cam_id = [0]
    cfg.lidar_s2_sim = True
    cfg.render_set     = {
        "fps"    : 24,
        "width"  : 640,
        "height" : 480
    }

    exec_node = MMK2ROS2(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    while rclpy.ok() and exec_node.running:
        exec_node.step(exec_node.target_control)

    exec_node.destroy_node()
    rclpy.shutdown()
    pubtopic_thread.join()
    spin_thread.join()
