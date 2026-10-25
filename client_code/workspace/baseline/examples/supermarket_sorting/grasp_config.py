"""Per-product grasp parameters for the supermarket sorting task.

Each product kind has its own gripper closure, deploy offset, reachable Z band
and grasp orientation. A single PickPlaceClient handles all 9 kinds.

Coordinate conventions (match supermarket_sorting_client.py):
  - deploy_offset: world (x,y,z) offset applied to the vision-estimated center
    to get the gripper target. +y points "into" the shelf (deeper), +z up.
  - grasp_rot: 3x3 rotation of the gripper in the arm-base frame.
  - reach_z_min/max: vertical band the arm can actually reach at the shelf.

Grasp strategy (the robot stops ~0.45 m in front of the slot, arm reaches in):
  * bottles / fruit (upright): R_ARM_SIDE  - side grasp, fingers horizontal
  * flat / soft items        : R_ARM_NATURAL - top approach
The deploy target is essentially the product centre (small offset) because the
base already stands at the reach limit.
"""
import numpy as np

IDENTITY = np.eye(3).tolist()

# Arm's natural gripper orientation (MMK2Kdl FK at the resting right-arm pose).
# IK-reachable at every shelf position; identity was not reachable for the arm.
R_ARM_NATURAL = [
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
]

# Side grasp (gripper horizontal, approach along +y in the footprint frame):
# used for upright bottles/fruit. IK-verified at L1/L2/L3.
R_ARM_SIDE = [
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]

# 90-degree rotation about X (to lay a flat item down / approach from top)
ROT_X_90 = [
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
]

# Common deploy target offset: the base stands ~0.45 m in front of the slot, so
# the gripper only needs a small pull-back from the product centre.
_OFF = [-0.011, -0.22, 0.02]

KIND_GRASP = {
    # --- bottle-like, upright, side grasp ---
    "kele": dict(
        grip_open=0.99, grip_close=0.08,
        deploy_offset=[-0.011, -0.220, 0.18], creep_stop_dy=0.035,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_SIDE, place_upright=True, approach="side",
        note="bottle; side grasp",
    ),
    "maidong": dict(  # small bottle
        grip_open=0.99, grip_close=0.10,
        deploy_offset=_OFF, creep_stop_dy=0.035,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_SIDE, place_upright=True, approach="side",
        note="bottle; side grasp",
    ),
    "kouxiangtang": dict(  # box/can, slightly wider
        grip_open=0.99, grip_close=0.18,
        deploy_offset=_OFF, creep_stop_dy=0.040,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_SIDE, place_upright=True, approach="side",
        note="can; side grasp",
    ),
    # --- round fruit, side grasp ---
    "pingguo": dict(  # apple
        grip_open=0.99, grip_close=0.45,
        deploy_offset=_OFF, creep_stop_dy=0.045,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_SIDE, place_upright=True, approach="side",
        note="apple; side grasp",
    ),
    "chengzi": dict(  # orange
        grip_open=0.99, grip_close=0.52,
        deploy_offset=_OFF, creep_stop_dy=0.045,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_SIDE, place_upright=True, approach="side",
        note="orange; side grasp",
    ),
    # --- flat/soft items, top approach ---
    "sanmingzhi": dict(  # sandwich
        grip_open=0.99, grip_close=0.62,
        deploy_offset=_OFF, creep_stop_dy=0.050,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_NATURAL, place_upright=False, approach="top",
        note="flat; top grasp",
    ),
    "shupian": dict(  # chips bag
        grip_open=0.99, grip_close=0.70,
        deploy_offset=_OFF, creep_stop_dy=0.055,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_NATURAL, place_upright=False, approach="top",
        note="flat; top grasp",
    ),
    "heweidao": dict(  # bread-ish
        grip_open=0.99, grip_close=0.60,
        deploy_offset=_OFF, creep_stop_dy=0.050,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_NATURAL, place_upright=False, approach="top",
        note="flat; top grasp",
    ),
    "zhijin": dict(  # tissue
        grip_open=0.99, grip_close=0.66,
        deploy_offset=_OFF, creep_stop_dy=0.050,
        reach_z_min=0.42, reach_z_max=1.35,
        grasp_rot=R_ARM_NATURAL, place_upright=False, approach="top",
        note="flat; top grasp",
    ),
}

# kinds present on a shelf slot for each aruco id (see retail_competition_layout.json)
# used to decide WHICH detection class to expect when picking a given target.
VALID_KINDS = list(KIND_GRASP.keys())


def get_grasp(kind: str) -> dict:
    if kind not in KIND_GRASP:
        raise KeyError(f"unknown kind '{kind}'; known: {VALID_KINDS}")
    g = dict(KIND_GRASP[kind])
    g["grasp_rot"] = np.array(g["grasp_rot"], dtype=float)
    g["deploy_offset"] = list(g["deploy_offset"])
    return g
