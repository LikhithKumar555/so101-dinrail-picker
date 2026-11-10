"""The DIN-rail scene: loading, resetting, and checking whether a block is placed correctly.

All coordinates are in the robot base frame (= MuJoCo world frame): x forward, y left, z up.
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from so101_sim.ik import solve_ik, top_down_pose
from so101_sim.kinematics import ARM_JOINTS, SO101Kinematics, make_transform, quat_to_rot

SCENE_XML = Path(__file__).resolve().parents[1] / "models" / "dinrail_scene.xml"

TABLE_Z = -0.08              # table surface (arm stands on an 8 cm riser)
RAIL_TOP_Z = TABLE_Z + 0.0075  # top of the TS35 flanges
BLOCK_DEPTH = 0.045          # along x (across the rail)
BLOCK_HEIGHT = 0.045         # along z


@dataclass(frozen=True)
class BlockSpec:
    name: str        # MJCF body / joint / geom name
    width: float     # along y (along the rail), metres
    supply_xy: tuple # nominal pick position on the table


# Supply positions are 6 cm apart so the opening jaw cannot hit a neighbouring block,
# and >= 2 cm clear of the rail end even with the +-1 cm randomisation.
BLOCKS = (
    BlockSpec("block_w5", 0.0052, (0.18, 0.07)),
    BlockSpec("block_w8", 0.0082, (0.18, 0.13)),
    BlockSpec("block_w12", 0.0122, (0.18, 0.19)),
)
SLOT_SITES = ("slot_0", "slot_1", "slot_2")

# Placement tolerances used to decide success (also used later for the real robot).
POS_TOL_ACROSS = 0.003   # m, across the rail (x): block must sit centred on the flanges
POS_TOL_ALONG = 0.005    # m, along the rail (y): distance to the target slot
HEIGHT_TOL = 0.002       # m, block must rest on the rail, not tilted on top of something
YAW_TOL = np.radians(5)  # block must be square to the rail
TILT_TOL = np.radians(5)  # block must stand upright


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    return model, mujoco.MjData(model)


def scene_kinematics() -> SO101Kinematics:
    """Kinematics read from the scene file, so it includes the scene's base placement."""
    return SO101Kinematics(SCENE_XML)


_READY_Q = None
# 'Park' pose: arm leaned back out of the overhead camera's view (for clean scene images).
PARK_Q = np.radians([0.0, -100.0, -95.0, 15.0, 0.0])


def ready_q() -> np.ndarray:
    """'Ready' pose: gripper pointing down, 8 cm above the table, in front of the supply area."""
    global _READY_Q
    if _READY_Q is None:
        res = solve_ik(scene_kinematics(), top_down_pose([0.20, 0.0, TABLE_Z + 0.08]))
        assert res.success, "ready pose not reachable"
        _READY_Q = res.q
    return _READY_Q.copy()


def body_pose(model, data, body: str) -> np.ndarray:
    """4x4 world pose of a body."""
    return make_transform(quat_to_rot(data.body(body).xquat), data.body(body).xpos)


def slot_pose(model, data, slot: str) -> np.ndarray:
    """4x4 pose a block centre must have when it is seated in `slot`."""
    T = make_transform(data.site(slot).xmat.reshape(3, 3), data.site(slot).xpos)
    T[2, 3] = RAIL_TOP_Z + BLOCK_HEIGHT / 2
    return T


def set_block_pose(model, data, block: str, xyz, yaw: float = 0.0):
    adr = model.joint(block).qposadr[0]
    data.qpos[adr:adr + 3] = xyz
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # yaw about z
    dadr = model.joint(block).dofadr[0]
    data.qvel[dadr:dadr + 6] = 0.0


def reset(model, data, rng: np.random.Generator | None = None,
          xy_jitter: float = 0.01, yaw_jitter_deg: float = 10.0, arm_pose: str = "ready"):
    """Arm to the 'ready' (or 'park') pose, blocks standing in the supply area (optionally randomised)."""
    mujoco.mj_resetData(model, data)
    q = ready_q() if arm_pose == "ready" else PARK_Q.copy()
    for j, qj in zip(ARM_JOINTS, q):
        data.qpos[model.joint(j).qposadr[0]] = qj
        data.ctrl[model.actuator(j).id] = qj
    for spec in BLOCKS:
        x, y = spec.supply_xy
        yaw = 0.0
        if rng is not None:
            x, y = x + rng.uniform(-xy_jitter, xy_jitter), y + rng.uniform(-xy_jitter, xy_jitter)
            yaw = np.radians(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        set_block_pose(model, data, spec.name, [x, y, TABLE_Z + BLOCK_HEIGHT / 2], yaw)
    mujoco.mj_forward(model, data)


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def placement_error(model, data, block: str, slot: str) -> dict:
    """How far a block is from being correctly seated in a slot."""
    T_b, T_s = body_pose(model, data, block), slot_pose(model, data, slot)
    d = T_s[:3, :3].T @ (T_b[:3, 3] - T_s[:3, 3])          # error in the slot frame
    R_rel = T_s[:3, :3].T @ T_b[:3, :3]
    yaw = wrap_angle(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
    yaw = min(abs(yaw), abs(wrap_angle(yaw + np.pi)))      # a box rotated 180 deg is the same
    tilt = np.arccos(np.clip(R_rel[2, 2], -1.0, 1.0))
    return {"across": d[0], "along": d[1], "height": d[2], "yaw": yaw, "tilt": tilt}


def is_placed(model, data, block: str, slot: str) -> bool:
    e = placement_error(model, data, block, slot)
    return (abs(e["across"]) < POS_TOL_ACROSS and abs(e["along"]) < POS_TOL_ALONG
            and abs(e["height"]) < HEIGHT_TOL and e["yaw"] < YAW_TOL and e["tilt"] < TILT_TOL)
