"""Jacobian and Jacobian-based inverse kinematics (damped least squares) for the SO-101.

Builds on kinematics.py. Conventions:
  * Poses are 4x4 homogeneous transforms in the world frame (T_world_tcp).
  * Twists / errors are 6-vectors [linear (3); angular (3)], both in the world frame.
  * The TCP approach axis (pointing out of the gripper, between the jaws) is the TCP x-axis.
"""
from dataclasses import dataclass

import numpy as np

from so101_sim.kinematics import SO101Kinematics, make_transform


# --------------------------------------------------------------------------
# Jacobian
# --------------------------------------------------------------------------
def geometric_jacobian(kin: SO101Kinematics, q: np.ndarray) -> np.ndarray:
    """6 x n geometric Jacobian of the TCP, expressed in the world frame.

    Column i (revolute joint i, unit axis z_i through point p_i, all in world):
        linear part   J_v,i = z_i x (p_tcp - p_i)
        angular part  J_w,i = z_i
    so that [v; w] = J(q) @ q_dot.
    """
    frames = kin.link_frames(q)
    p_tcp = (frames[-1] @ kin.T_last_tcp)[:3, 3]
    J = np.zeros((6, kin.n_joints))
    for i, (T_i, link) in enumerate(zip(frames, kin.links)):
        z_i = T_i[:3, :3] @ link.axis        # joint axis rotated into the world frame
        p_i = T_i[:3, 3]                     # a point on the joint axis (joint at body origin)
        J[:3, i] = np.cross(z_i, p_tcp - p_i)
        J[3:, i] = z_i
    return J


def manipulability(J_pos: np.ndarray) -> float:
    """Yoshikawa manipulability sqrt(det(J J^T)); goes to 0 at a singularity."""
    return float(np.sqrt(max(np.linalg.det(J_pos @ J_pos.T), 0.0)))


# --------------------------------------------------------------------------
# Orientation helpers
# --------------------------------------------------------------------------
def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z), numerically robust (Shepperd)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def orientation_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, world frame) that rotates R onto R_target."""
    q = rot_to_quat(R_target @ R.T)
    if q[0] < 0:                              # q and -q are the same rotation; take the short way
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, q[0])
    return v / s * angle


def pose_error(T_target: np.ndarray, T: np.ndarray) -> np.ndarray:
    """6-vector [position error; orientation error] in the world frame."""
    return np.concatenate([T_target[:3, 3] - T[:3, 3],
                           orientation_error(T_target[:3, :3], T[:3, :3])])


def top_down_pose(position: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    """Grasp pose with the gripper pointing straight down, jaws rotated by `yaw` about vertical.

    TCP x-axis (approach) = world -z; TCP y-axis = horizontal at angle `yaw`.
    """
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    z = np.cross(x, y)
    return make_transform(np.column_stack([x, y, z]), np.asarray(position, dtype=float))


# --------------------------------------------------------------------------
# Inverse kinematics: damped least squares (Levenberg-Marquardt style)
# --------------------------------------------------------------------------
@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    pos_error: float   # metres
    rot_error: float   # radians


def solve_ik(kin: SO101Kinematics, T_target: np.ndarray, q0: np.ndarray | None = None, *,
             rot_weight: float = 0.3, damping: float = 0.05, max_step: float = 0.2,
             max_iters: int = 300, pos_tol: float = 1e-4, rot_tol: float = 1e-3,
             restarts: int = 10, seed: int = 0) -> IKResult:
    """Find joint angles q with FK(q) ~= T_target.

    Each iteration solves the linearised problem  J dq = e  with damping:
        dq = J^T (J J^T + lambda^2 I)^-1 e
    which behaves like the pseudo-inverse far from singularities and stays bounded near them.
    rot_weight scales orientation (rad) against position (m); set it to 0 for position-only IK.
    If a solve does not converge (local minimum / joint limits), retry from a random start.
    """
    lo, hi = kin.joint_limits[:, 0], kin.joint_limits[:, 1]
    W = np.diag([1.0, 1.0, 1.0] + [rot_weight] * 3)
    rng = np.random.default_rng(seed)
    q_start = np.zeros(kin.n_joints) if q0 is None else np.asarray(q0, dtype=float)

    best = None
    for attempt in range(restarts + 1):
        q = np.clip(q_start if attempt == 0 else rng.uniform(lo, hi), lo, hi)
        for it in range(1, max_iters + 1):
            e = pose_error(T_target, kin.forward_kinematics(q))
            pos_err, rot_err = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            converged = pos_err < pos_tol and (rot_weight == 0 or rot_err < rot_tol)
            if converged:
                break
            J = W @ geometric_jacobian(kin, q)
            ew = W @ e
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), ew)
            step = np.max(np.abs(dq))
            if step > max_step:                      # limit how far one linear step can go
                dq *= max_step / step
            q = np.clip(q + dq, lo, hi)              # respect joint limits
        result = IKResult(q, converged, it, pos_err, rot_err)
        if converged:
            return result
        if best is None or pos_err + rot_weight * rot_err < best.pos_error + rot_weight * best.rot_error:
            best = result
    return best
