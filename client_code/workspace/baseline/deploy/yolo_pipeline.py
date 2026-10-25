#!/usr/bin/env python3
"""YOLO 数据集自动标注 + 训练 + 导出 + 推理 完整流水线 (DISCOVERSE / MuJoCo + 3DGS).

核心是 `YoloLabeler` —— 可直接嵌入你现有的仿真循环:每渲染一帧调用一次
`label_frame()`,输入该帧 RGB(和可选深度),返回该帧所有可见物品的
YOLO 格式标注(归一化 x_center y_center width height)。

投影约定(已在场景中实测验证,误差 ~0.2px):
    R    = diag(1,-1,-1) @ mj_data.cam_xmat[cam_id]   # MuJoCo -> 图像系
    pc   = R.T @ (p_world - mj_data.cam_xpos[cam_id])
    u    = fx * pc[0]/pc[2] + cx ;  v = fy * pc[1]/pc[2] + cy
内参:  fx = fy = (H/2) / tan(fovy/2), cx=(W-1)/2, cy=(H-1)/2, fovy 来自 <camera>。

子命令:
  gen    在仿真里巡航 + 随机化,生成标注图片(images/labels + data.yaml + 划分)
  train  用 Ultralytics YOLOv8s 训练
  export 导出 ONNX
  infer  加载模型对一帧做推理

嵌入 DISCOVERSE 现有循环:
    from yolo_pipeline import YoloLabeler
    labeler = YoloLabeler(env.mj_model, env.mj_data, cam_id=0,
                          width=W, height=H,
                          classes=["kele", "sanmingzhi", ...],
                          sizes={"kele": (0.04, 0.04, 0.12), ...})
    # 在你的主循环里,每渲染一帧:
    env.render()
    labels = labeler.label_frame(env.img_rgb_obs_s[0],
                                 env.img_depth_obs_s[0])  # 深度 mm -> 米,自动
    # labels = [(class_id, x_c, y_c, w, h) 归一化];写 YOLO txt 即可
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # 允许无 mujoco 环境下只跑 train/export
    mujoco = None


# ---------------------------------------------------------------------------
# 投影 / 标注器
# ---------------------------------------------------------------------------
class YoloLabeler:
    """把当前仿真帧里所有物品的 3D 包围盒投影成 YOLO 标注。

    model / data: MuJoCo 的 MjModel / MjData(或 DISCOVERSE env 的 mj_model/mj_data)。
    class_bodies: 有序 dict {类别名: [body 名, ...]},同一个类可以有多个 body
                  (比如同类商品有 5 个实例)。YOLO 的 class_id = 该类别在 dict 中的下标。
    sizes:        {body 名: (半长, 半宽, 半高)} 米;缺省用 0.05 立方体。
    """

    def __init__(self, model, data, cam_id, width, height,
                 class_bodies, sizes=None, min_side_px=16, depth_mm=True):
        self.m, self.d = model, data
        self.cam_id = cam_id
        self.W, self.H = width, height
        self.class_names = list(class_bodies.keys())
        self.cls_body_ids = []
        for cls in self.class_names:
            ids = []
            for bname in class_bodies[cls]:
                ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname))
            self.cls_body_ids.append(ids)
        self.sizes = sizes or {}
        self.min_side_px = min_side_px
        self.depth_mm = depth_mm
        fovy = float(model.cam_fovy[cam_id])
        self.fx = self.fy = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0

    # -- 相机 --
    def cam_R(self):
        return np.diag([1.0, -1.0, -1.0]) @ self.d.cam_xmat[self.cam_id].reshape(3, 3)

    def cam_pos(self):
        return self.d.cam_xpos[self.cam_id]

    def project(self, p_world):
        """世界坐标 -> (u, v, depth)。在相机后方/过近返回 None。"""
        pc = self.cam_R().T @ (p_world - self.cam_pos())
        z = pc[2]
        if z <= 0.05:
            return None
        u = self.fx * pc[0] / z + self.cx
        v = self.fy * pc[1] / z + self.cy
        return u, v, z

    def item_box_corners(self, body_id):
        """物品在世界系下的 8 个包围盒顶点:中心 xpos + 旋转 xmat @ 半尺寸。"""
        half = (0.05, 0.05, 0.05)
        nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if nm in self.sizes:
            half = self.sizes[nm]
        pos = self.d.xpos[body_id]
        rot = self.d.xmat[body_id].reshape(3, 3)
        hx, hy, hz = half
        base = np.array([[hx, hy, hz], [-hx, hy, hz], [hx, -hy, hz], [-hx, -hy, hz],
                         [hx, hy, -hz], [-hx, hy, -hz], [hx, -hy, -hz], [-hx, -hy, -hz]])
        return pos + base @ rot.T

    def label_frame(self, rgb, depth=None):
        """输入一帧 BGR/RGB(和可选深度图),返回 [(cls_id, x_center, y_center, w, h) 归一化]。

        跳过: 物品在相机后方、框太小、框越界、被遮挡(depth 给出时)。"""
        H, W = rgb.shape[:2]
        out = []
        for cid, body_ids in enumerate(self.cls_body_ids):
            for bid in body_ids:
                if bid < 0:
                    continue
                corners = self.item_box_corners(bid)
                pts = [self.project(c) for c in corners]
                valid = [p for p in pts if p is not None]
                if len(valid) < 4:
                    continue  # 大部分顶点在相机后方
                us = [p[0] for p in valid]
                vs = [p[1] for p in valid]
                x0, x1 = min(us), max(us)
                y0, y1 = min(vs), max(vs)
                bw, bh = x1 - x0, y1 - y0
                if bw < self.min_side_px or bh < self.min_side_px:
                    continue  # 太小(太远)
                if x0 < -50 or y0 < -50 or x1 > W + 50 or y1 > H + 50:
                    continue  # 严重越界
                # 裁剪到图像内
                cx0, cy0 = max(0, x0), max(0, y0)
                cx1, cy1 = min(W - 1, x1), min(H - 1, y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                # 遮挡检测:中心像素深度 vs 物品中心期望深度
                cu, cv = (cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0
                if depth is not None:
                    pc = self.project(np.asarray(self.d.xpos[bid], float))
                    if pc is None:
                        continue
                    dz = self._depth_at(depth, int(cu), int(cv))
                    if dz <= 0.0 or abs(dz - pc[2]) > 0.20:
                        continue  # 中心被别的东西挡住
                out.append((cid,
                            (cx0 + cx1) / 2.0 / W,
                            (cy0 + cy1) / 2.0 / H,
                            (cx1 - cx0) / W,
                            (cy1 - cy0) / H))
        return out

    def _depth_at(self, depth, u, v):
        """取深度图(u,v)附近中值,返回米。"""
        h, w = depth.shape[:2]
        y0, y1 = max(0, v - 2), min(h, v + 3)
        x0, x1 = max(0, u - 2), min(w, u + 3)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0.0
        m = float(np.median(valid))
        return m * 1e-3 if self.depth_mm else m


# ---------------------------------------------------------------------------
# 数据集生成
# ---------------------------------------------------------------------------
def _random_background(rng, h, w):
    kind = rng.integers(0, 4)
    if kind == 0:
        return np.ones((h, w, 3), np.uint8) * rng.integers(0, 256, size=3).astype(np.uint8)
    elif kind == 1:
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif kind == 2:
        yy, xx = np.mgrid[0:h, 0:w]
        ang = rng.uniform(0, np.pi)
        t = (np.cos(ang) * xx + np.sin(ang) * yy) * rng.uniform(0.5, 2.0) / max(h, 1)
        col = rng.integers(0, 256, size=3).astype(np.uint8)
        col2 = rng.integers(0, 256, size=3).astype(np.uint8)
        a = (t - t.min()) / (t.max() - t.min() + 1e-6)
        return (col * (1 - a[..., None]) + col2 * a[..., None]).astype(np.uint8)
    else:
        return np.full((h, w, 3), 128, np.uint8)


def _composite_background(rgb, bg):
    """简单前景抠像:很暗的背景像素换成随机背景(DISCOVERSE 3DGS 前景是亮色)。"""
    fg = rgb.max(axis=2) > 30
    out = bg.copy()
    out[fg] = rgb[fg]
    return out


def _auto_sizes(m, d, body_names):
    """从每个 body 的碰撞几何(capsule/box/...)自动算 3D 包围盒半尺寸(米)。"""
    sizes = {}
    for bname in body_names:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        bpos = d.xpos[bid]
        brot = d.xmat[bid].reshape(3, 3)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid:
                continue
            gp = d.geom_xpos[g]
            gr = d.geom_xmat[g].reshape(3, 3)
            s = m.geom_size[g]
            gt = m.geom_type[g]
            if gt == mujoco.mjtGeom.mjGEOM_BOX:
                pts = np.array([[a, b, c] for a in (-s[0], s[0])
                                for b in (-s[1], s[1]) for c in (-s[2], s[2])])
            elif gt in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                r, L = s[0], s[1]
                pts = np.array([[a, b, c] for a in (-r, r) for b in (-r, r)
                                for c in (-(L + r), L + r)])
            elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = s[0]
                pts = np.array([[a, b, c] for a in (-r, r) for b in (-r, r)
                                for c in (-r, r)])
            else:
                continue
            for p in pts:
                w = gp + gr @ p
                l = brot.T @ (w - bpos)
                lo = np.minimum(lo, l)
                hi = np.maximum(hi, l)
        if not np.any(np.isinf(lo)):
            sizes[bname] = tuple(((hi - lo) / 2.0).round(4))
        else:
            sizes[bname] = (0.05, 0.05, 0.05)
    return sizes


def _write_yolo_txt(path, labels):
    with open(path, "w") as f:
        for cid, xc, yc, bw, bh in labels:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _cruise_waypoints(center, radius, n):
    return [(center[0] + radius * math.cos(2 * math.pi * i / n),
             center[1] + radius * math.sin(2 * math.pi * i / n),
             2 * math.pi * i / n + math.pi) for i in range(n)]


def gen(args):
    import cv2
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "images" / "test").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "test").mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    render_env = None
    if args.env:
        # 用 DISCOVERSE 环境渲染(3DGS 商品真实外观),而不是内置渲染器
        import importlib
        mod = importlib.import_module(args.env)
        cfg = mod.build_config()
        from discoverse.robots_env.mmk2_base import MMK2Base
        env = MMK2Base(cfg)
        env.reset()
        m, d = env.mj_model, env.mj_data
        render_env = env
        cam_names = list(getattr(env, "camera_names", []) or
                         [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                          for i in range(m.ncam)])
        print(f"[gen] DISCOVERSE env up; cameras: {cam_names}")

    # 相机:默认第一个相机,或用名字
    if args.cam:
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
        if cam_id < 0:
            raise SystemExit(f"camera '{args.cam}' not found")
    else:
        cam_id = 0
    cam_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    print(f"cameras: {cam_names}; using id {cam_id}")

    # 类别:优先用 --class-map {body: class} 聚合成类;否则自动取 freejoint body 作为类
    class_bodies = {}
    if args.class_map:
        cmap = json.loads(Path(args.class_map).read_text())
        for body, cls in cmap.items():
            class_bodies.setdefault(cls, []).append(body)
    elif args.env and args.layout:
        # 从布局自动建 map:item_run_<hash>_NN -> product_NN -> object_kind
        layout = json.loads(Path(args.layout).read_text())
        kind_of = {s["body"]: s["object_kind"] for s in layout}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])
                if nm and "item_run" in nm:
                    idx = int(nm.rsplit("_", 1)[-1])
                    kind = kind_of.get(f"product_{idx:03d}")
                    if kind:
                        class_bodies.setdefault(kind, []).append(nm)
    else:
        cam_body = m.cam_bodyid[cam_id]
        robot_root = cam_body
        while m.body_parentid[robot_root] != 0:
            robot_root = m.body_parentid[robot_root]
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                bid = m.jnt_bodyid[j]
                if bid == robot_root:
                    continue  # 机器人本体,不是待检测物品
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
                if nm and nm not in class_bodies:
                    class_bodies[nm] = [nm]
    if not class_bodies:
        raise SystemExit("no classes found (use --classes/--class-map or check the XML)")
    classes = list(class_bodies.keys())
    print(f"classes({len(classes)}):", classes)

    # 尺寸:自动从碰撞几何算;--sizes 里的显式覆盖
    mujoco.mj_forward(m, d)
    all_bodies = [b for bs in class_bodies.values() for b in bs]
    sizes = _auto_sizes(m, d, all_bodies)
    if args.sizes:
        for item in args.sizes.split(","):
            name, _, dim = item.partition("=")
            sizes[name] = tuple(float(x) for x in dim.split("x"))
    print("[gen] sizes:", {k: sizes.get(k) for k in all_bodies[:3]}, "...")

    labeler = YoloLabeler(m, d, cam_id, args.width, args.height, class_bodies, sizes)
    renderer = mujoco.Renderer(m, args.height, args.width)

    # 随机化:物品 free joint 的 qpos 在允许范围内抖动(位移+偏航)
    item_qpos = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            item_qpos[m.jnt_qposadr[j]] = d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7].copy()

    total = 0
    split_of = {"train": 0, "val": 0, "test": 0}

    # ---- 巡航:绕物品中心转圈,每步 P 控制器驱动底盘 free joint ----
    center = np.mean([labeler.d.xpos[b] for ids in labeler.cls_body_ids for b in ids if b >= 0],
                     axis=0) if any(labeler.cls_body_ids) else np.zeros(3)
    base_adr = None
    if args.base:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.base)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == bid:
                base_adr = m.jnt_qposadr[j]
                break
    if base_adr is None:
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                base_adr = m.jnt_qposadr[j]
                break
    print(f"[gen] cruising around item centroid {np.round(center,2)}; base freejoint adr={base_adr}")
    if args.waypoints_file:
        waypoints = [tuple(w) for w in json.loads(Path(args.waypoints_file).read_text())]
    else:
        waypoints = _cruise_waypoints(center[:2], args.radius, args.waypoints)
    print(f"[gen] {len(waypoints)} viewpoints; base freejoint adr={base_adr}")

    def base_pose():
        q = d.qpos[base_adr:base_adr + 7]
        yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        return q[0], q[1], yaw

    def set_base(x, y, yaw):
        d.qpos[base_adr:base_adr + 3] = [x, y, 0.0]
        d.qpos[base_adr + 3:base_adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    for step in range(args.frames):
        # 随机化物品位姿(每 args.jitter_every 帧)
        if args.jitter and step % args.jitter_every == 0:
            for adr, base in item_qpos.items():
                d.qpos[adr:adr + 3] = base[0:3] + rng.uniform(-args.jitter, args.jitter, 3) \
                    * np.array([1, 1, 0])
                dyaw = rng.uniform(-0.5, 0.5)
                d.qpos[adr + 3:adr + 7] = [math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)]
        # 巡航:把底盘摆到当前路点(仿真步长很小,直接放置即可获得多视角)
        tx, ty, tyaw = waypoints[step % len(waypoints)]
        set_base(tx + rng.uniform(-args.pose_jit, args.pose_jit),
                 ty + rng.uniform(-args.pose_jit, args.pose_jit),
                 tyaw + rng.uniform(-args.pose_jit, args.pose_jit))
        # 头部俯仰扫掠,让不同高度的物品都能入镜
        if args.head_pitch_joint:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, args.head_pitch_joint)
            if jid >= 0:
                ph = args.head_pitch_lo + (args.head_pitch_hi - args.head_pitch_lo) * 0.5 * \
                    (1 + math.sin(2 * math.pi * step / args.head_pitch_period))
                d.qpos[m.jnt_qposadr[jid]] = ph
        mujoco.mj_forward(m, d)
        if render_env is not None:
            render_env.render()
            rgb = render_env.img_rgb_obs_s[cam_id]
            rgb = np.ascontiguousarray(rgb[..., ::-1])  # RGB -> BGR
        else:
            renderer.update_scene(d, camera=cam_id)
            rgb = renderer.render()

        labels = labeler.label_frame(rgb, None)
        if len(labels) == 0:
            continue
        split = "train" if total % 10 < 8 else ("val" if total % 10 < 9 else "test")
        name = f"f{total:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), rgb)
        _write_yolo_txt(out / "labels" / split / f"{name}.txt", labels)
        total += 1
        split_of[split] += 1
        if total >= args.target:
            break

    # data.yaml
    data_yaml = (f"path: {out.resolve()}\n"
                 f"train: images/train\nval: images/val\n"
                 f"nc: {len(classes)}\nnames: {json.dumps(classes)}\n")
    (out / "data.yaml").write_text(data_yaml)
    print(f"[gen] done: {total} images -> {split_of}")
    print(f"[gen] data.yaml -> {out / 'data.yaml'}")


# ---------------------------------------------------------------------------
# 训练 / 导出 / 推理
# ---------------------------------------------------------------------------
def train(args):
    import torch
    # 兼容旧版 yolov8s.pt / AMP 检查的 yolov8n.pt 权重(PyTorch>=2.6 默认 weights_only=True)
    _orig = torch.load
    def _compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _compat
    from ultralytics import YOLO
    model = YOLO(args.weights)
    try:
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, patience=args.patience,
                    project=args.project, name=args.name)
    finally:
        torch.load = _orig
    best = os.path.join(args.project, args.name, "weights", "best.pt")
    metrics = model.val()  # 验证集 mAP50 / mAP50-95
    print(f"[train] best.pt -> {best}")
    print(f"[train] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


def export(args):
    import torch
    _orig = torch.load
    def _compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _compat
    from ultralytics import YOLO
    model = YOLO(args.weights)
    torch.load = _orig
    model.export(format="onnx", imgsz=args.imgsz, dynamic=args.dynamic)
    onnx = str(Path(args.weights).with_suffix(".onnx"))
    print(f"[export] onnx -> {onnx}")


def infer(args):
    import cv2
    import torch
    _orig = torch.load
    def _compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _compat
    from ultralytics import YOLO
    model = YOLO(args.weights)  # 也支持 ONNX:自动用 onnxruntime/OpenCV DNN
    torch.load = _orig
    if args.image:
        results = model.predict(args.image, conf=args.conf, imgsz=args.imgsz, save=True)
        r = results[0]
        for box in r.boxes:
            print(f"{r.names[int(box.cls)]} conf={box.conf[0]:.3f} xyxy={box.xyxy[0].tolist()}")
    elif args.xml:
        m = mujoco.MjModel.from_xml_path(args.xml)
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, args.height, args.width)
        renderer.update_scene(d, camera=0)
        rgb = renderer.render()
        results = model.predict(rgb, conf=args.conf, imgsz=args.imgsz, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                print(f"{r.names[int(box.cls)]} conf={box.conf[0]:.3f} "
                      f"box=({x1},{y1})-({x2},{y2})")
    else:
        raise SystemExit("infer needs --image or --xml")


def main():
    ap = argparse.ArgumentParser(prog="yolo_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="在仿真里生成标注数据")
    g.add_argument("--xml", required=True, help="场景 MJCF/运行时 XML 路径")
    g.add_argument("--env", default="", help="DISCOVERSE 环境模块(有 build_config),用它渲染 3DGS 外观")
    g.add_argument("--layout", default="", help="布局 JSON(自动建 body->类别映射)")
    g.add_argument("--out", default="yolo_data")
    g.add_argument("--cam", default=None, help="相机名(默认第一个)")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--classes", default="", help="逗号分隔的类别名(默认取 freejoint body)")
    g.add_argument("--class-map", default="", help="JSON {body名: 类别名},把多个 body 归为一类")
    g.add_argument("--sizes", default="", help="半尺寸,如 kele=0.04x0.04x0.12")
    g.add_argument("--frames", type=int, default=800, help="仿真步数上限")
    g.add_argument("--target", type=int, default=600, help="目标图片数")
    g.add_argument("--jitter", type=float, default=0.03, help="物品位姿随机抖动幅度(m)")
    g.add_argument("--jitter-every", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--radius", type=float, default=1.2, help="巡航半径(m)")
    g.add_argument("--waypoints", type=int, default=8, help="巡航路点数")
    g.add_argument("--waypoints-file", default="", help="路点 JSON [[x,y,yaw],...]")
    g.add_argument("--pose-jit", type=float, default=0.05, help="路点处底盘位姿抖动")
    g.add_argument("--every", type=int, default=5, help="每 N 步采一帧")
    g.add_argument("--base", default="", help="机器人底盘 body 名(默认第一个 freejoint)")
    g.add_argument("--head-pitch-joint", default="head_pitch_joint",
                   help="头部俯仰关节名(不存在则跳过)")
    g.add_argument("--head-pitch-lo", type=float, default=-1.0)
    g.add_argument("--head-pitch-hi", type=float, default=-0.2)
    g.add_argument("--head-pitch-period", type=int, default=120)
    g.set_defaults(fn=gen)

    t = sub.add_parser("train", help="YOLOv8s 训练")
    t.add_argument("--data", required=True)
    t.add_argument("--weights", default="yolov8s.pt")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--device", default="0")
    t.add_argument("--patience", type=int, default=100)
    t.add_argument("--project", default="runs")
    t.add_argument("--name", default="yolo")
    t.set_defaults(fn=train)

    e = sub.add_parser("export", help="导出 ONNX")
    e.add_argument("--weights", required=True)
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--dynamic", action="store_true")
    e.set_defaults(fn=export)

    i = sub.add_parser("infer", help="推理")
    i.add_argument("--weights", required=True)
    i.add_argument("--image", default=None, help="图片路径")
    i.add_argument("--xml", default=None, help="或渲染一帧的场景 XML")
    i.add_argument("--width", type=int, default=640)
    i.add_argument("--height", type=int, default=480)
    i.add_argument("--conf", type=float, default=0.25)
    i.add_argument("--imgsz", type=int, default=640)
    i.set_defaults(fn=infer)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()