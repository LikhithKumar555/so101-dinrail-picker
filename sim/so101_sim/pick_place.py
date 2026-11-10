"""Scripted (classic) pick-and-place of terminal blocks onto the DIN rail.

Pipeline per block:  ready -> above block -> straight down -> close -> straight up
                     -> carry at safe height -> above slot -> straight down -> open -> straight up
Every Cartesian segment is a straight TCP line, turned into joint angles with our own IK.
The block poses come from the simulator here; in Phase 4 they will come from the camera.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101_sim.ik import solve_ik, top_down_pose
from so101_sim.kinematics import ARM_JOINTS, TCP_SITE
from so101_sim.scene import (BLOCK_HEIGHT, RAIL_TOP_Z, TABLE_Z, BlockSpec, body_pose, is_placed,
                             scene_kinematics, slot_pose, wrap_angle)

# ---- Gripper (measured in the model, see step 2.4) ----
# Jaw-tip gap [mm] measured at several gripper angles [deg]; interpolate between them.
_GAP_TABLE_DEG = np.array([-9.7, -1.3, 7.1, 15.5, 23.9, 32.4])
_GAP_TABLE_MM = np.array([1.8, 13.0, 24.1, 35.0, 45.2, 54.3])
GRIPPER_CLOSED = np.radians(-10.0) # fully closed: the servo squeezes until its torque limit
GRIPPER_PARKED = np.radians(20.0)  # ~40 mm, used while moving without a block
PRESHAPE_MARGIN = 0.010            # m: open only to block width + this before grasping/releasing
FIXED_JAW_INNER = 0.021            # m: inner face of the fixed jaw, -z_tcp from the TCP (measured in sim)
JAW_CLEARANCE = 0.004              # m: gap between fixed jaw and block while descending
RELEASE_BACKOFF = 0.002            # m: move the fixed jaw off the block before lifting


def gripper_angle_for_gap(gap: float) -> float:
    """Gripper joint angle [rad] that gives a jaw-tip gap of `gap` metres (table lookup)."""
    return np.radians(np.interp(gap * 1000, _GAP_TABLE_MM, _GAP_TABLE_DEG))

# ---- Motion parameters ----
GRASP_DEPTH = 0.012                # TCP this far below the block's top face when grasping
APPROACH_HEIGHT = 0.06             # straight-line approach / retreat distance
CARRY_Z = TABLE_Z + 0.10           # safe transport height of the TCP
CART_SPEED = 0.10                  # m/s along straight TCP lines
WAYPOINT_STEP = 0.005              # m between IK waypoints on a straight line


@dataclass
class SimRobot:
    """Thin wrapper that drives the simulated SO-101 through its position servos."""
    model: mujoco.MjModel
    data: mujoco.MjData
    on_step: callable = None                      # optional hook (viewer sync, video recording)
    kin: object = field(default_factory=scene_kinematics)

    def __post_init__(self):
        m = self.model
        self.qadr = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.act = [m.actuator(j).id for j in ARM_JOINTS]
        self.grip_act = m.actuator("gripper").id
        self.tcp_site = m.site(TCP_SITE).id
        self.q_cmd = self.data.ctrl[self.act].copy()

    # -- low level --
    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step(self.model, self.data)

    def wait(self, seconds: float):
        self.step(int(seconds / self.model.opt.timestep))

    def tcp_pose_cmd(self) -> np.ndarray:
        """TCP pose of the *commanded* joints (what the planner believes)."""
        return self.kin.forward_kinematics(self.q_cmd)

    # -- motions --
    def follow_joint_path(self, path: list, duration: float):
        """Track a list of joint vectors with a smooth (zero start/end velocity) time law."""
        path = [self.q_cmd.copy()] + list(path)
        n_steps = max(1, int(duration / self.model.opt.timestep))
        for k in range(1, n_steps + 1):
            s = k / n_steps
            s = 3 * s**2 - 2 * s**3                   # smoothstep
            x = s * (len(path) - 1)
            i = min(int(x), len(path) - 2)
            q = path[i] + (x - i) * (path[i + 1] - path[i])
            self.data.ctrl[self.act] = q
            self.step()
        self.q_cmd = path[-1].copy()

    def move_line(self, T_goal: np.ndarray, speed: float = CART_SPEED) -> bool:
        """Straight-line TCP motion (position + yaw) solved with IK at closely spaced waypoints."""
        T_start = self.tcp_pose_cmd()
        p0, p1 = T_start[:3, 3], T_goal[:3, 3]
        yaw0 = np.arctan2(-T_start[0, 1], T_start[1, 1])   # recover yaw from the TCP y-axis
        yaw1 = np.arctan2(-T_goal[0, 1], T_goal[1, 1])
        dyaw = wrap_angle(yaw1 - yaw0)
        dist = np.linalg.norm(p1 - p0)
        n = max(2, int(np.ceil(max(dist / WAYPOINT_STEP, abs(dyaw) / 0.05))))
        path, q = [], self.q_cmd.copy()
        for s in np.linspace(0, 1, n + 1)[1:]:
            res = solve_ik(self.kin, top_down_pose(p0 + s * (p1 - p0), yaw0 + s * dyaw), q0=q,
                           restarts=0)
            if not res.success:
                return False
            q = res.q
            path.append(q)
        duration = max(dist / speed, abs(dyaw) / 1.0, 0.3)
        self.follow_joint_path(path, duration)
        return True

    def move_gripper(self, angle: float, duration: float = 0.6):
        start = self.data.ctrl[self.grip_act]
        n = int(duration / self.model.opt.timestep)
        for k in range(1, n + 1):
            self.data.ctrl[self.grip_act] = start + (angle - start) * k / n
            self.step()


def grasp_yaw(block_yaw: float, current_yaw: float) -> float:
    """Jaws must close along the block's width (its local y); pick the equivalent yaw
    (two choices, 180 deg apart) that needs the least wrist rotation."""
    candidates = [wrap_angle(block_yaw + np.pi / 2), wrap_angle(block_yaw - np.pi / 2)]
    return min(candidates, key=lambda y: abs(wrap_angle(y - current_yaw)))


def tcp_for_block_center(center: np.ndarray, yaw: float, width: float, clearance: float) -> np.ndarray:
    """TCP pose that puts the block centre between the jaws, against the fixed jaw.

    The fixed jaw's inner face is FIXED_JAW_INNER along -z_tcp from the TCP. The block face on
    that side must be `clearance` short of it, so the block centre sits at
    (FIXED_JAW_INNER - width/2 - clearance) along -z_tcp, i.e. TCP = centre + that * z_tcp.
    """
    T = top_down_pose(center, yaw)
    z_tcp = T[:3, 2]
    T[:3, 3] = center + (FIXED_JAW_INNER - width / 2 - clearance) * z_tcp
    return T


def pick_and_place(robot: SimRobot, block: BlockSpec, slot: str) -> tuple[bool, str]:
    """Run one pick-and-place. Returns (success, message)."""
    m, d = robot.model, robot.data
    T_block = body_pose(m, d, block.name)               # Phase 4: comes from the camera instead
    block_yaw = np.arctan2(T_block[1, 0], T_block[0, 0])
    cur_yaw = np.arctan2(-robot.tcp_pose_cmd()[0, 1], robot.tcp_pose_cmd()[1, 1])
    yaw = grasp_yaw(block_yaw, cur_yaw)

    # ---- pick ----
    center = T_block[:3, 3].copy()
    center[2] = TABLE_Z + BLOCK_HEIGHT - GRASP_DEPTH     # grasp height = top face - GRASP_DEPTH
    T_grasp = tcp_for_block_center(center, yaw, block.width, JAW_CLEARANCE)
    T_above = T_grasp.copy()
    T_above[2, 3] = CARRY_Z

    # Pre-shape: open only a little wider than the block, so the swinging jaw
    # cannot hit neighbouring blocks (we know the width from the block type).
    preshape = gripper_angle_for_gap(block.width + PRESHAPE_MARGIN)
    if not robot.move_line(T_above):
        return False, "cannot reach above block"
    robot.move_gripper(preshape, 0.4)
    if not robot.move_line(T_grasp, speed=0.05):
        return False, "cannot reach grasp pose"
    robot.move_gripper(GRIPPER_CLOSED, 0.6)
    robot.wait(0.3)
    T_lift = T_grasp.copy()
    T_lift[2, 3] = CARRY_Z
    if not robot.move_line(T_lift, speed=0.05):
        return False, "cannot lift"

    # ---- place ----
    T_slot = slot_pose(m, d, slot)
    slot_yaw = np.arctan2(T_slot[1, 0], T_slot[0, 0])
    yaw_place = grasp_yaw(slot_yaw, yaw)
    center = T_slot[:3, 3].copy()
    center[2] = RAIL_TOP_Z + BLOCK_HEIGHT - GRASP_DEPTH + 0.001   # release 1 mm above the rail
    T_place = tcp_for_block_center(center, yaw_place, block.width, 0.0)  # block now touches fixed jaw
    T_above_slot = T_place.copy()
    T_above_slot[2, 3] = CARRY_Z
    if not robot.move_line(T_above_slot):
        return False, "cannot reach above slot"
    if not robot.move_line(T_place, speed=0.05):
        return False, "cannot reach place pose"
    # Release: open a little, slide the fixed jaw off the block face, then lift straight up.
    robot.move_gripper(preshape, 0.8)
    robot.wait(0.2)
    T_backoff = robot.tcp_pose_cmd()
    T_backoff[:3, 3] -= RELEASE_BACKOFF * T_backoff[:3, 2]
    robot.move_line(T_backoff, speed=0.01)
    T_retreat = T_backoff.copy()
    T_retreat[2, 3] = CARRY_Z
    robot.move_line(T_retreat, speed=0.03)
    robot.move_gripper(GRIPPER_PARKED, 0.3)
    robot.wait(0.5)

    ok = is_placed(m, d, block.name, slot)
    return ok, "placed" if ok else "block not within tolerance"
