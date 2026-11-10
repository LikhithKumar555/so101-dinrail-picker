"""Forward kinematics of the SO-101 arm, written from scratch with NumPy.

MuJoCo is only used to READ the kinematic parameters (link offsets, joint axes,
TCP offset) from the MJCF. All the math is done here; tests compare it to MuJoCo.
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "robotstudio_so101"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_SITE = "gripperframe"


# --------------------------------------------------------------------------
# Rotation / transform helpers
# --------------------------------------------------------------------------
def quat_to_rot(quat: np.ndarray) -> np.ndarray:
    """Quaternion in MuJoCo order (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector -> skew-symmetric matrix [v]x, so that skew(a) @ b == np.cross(a, b)."""
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,   0.0, -x],
        [-y,  x,   0.0],
    ])


def axis_angle_to_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation of `theta` rad about `axis` (Rodrigues): R = I + sin(t)K + (1 - cos(t))K^2."""
    K = skew(np.asarray(axis, dtype=float) / np.linalg.norm(axis))
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform [[R, p], [0, 0, 0, 1]]."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


# --------------------------------------------------------------------------
# Robot model
# --------------------------------------------------------------------------
@dataclass
class Link:
    body: str                  # MJCF body that this joint moves
    T_parent_body: np.ndarray  # fixed 4x4 offset parent body -> this body (at q = 0)
    axis: np.ndarray           # joint axis, expressed in this body's frame


class SO101Kinematics:
    """Serial kinematic chain: world -> base -> 5 arm joints -> TCP."""

    def __init__(self, xml_path: Path = MODEL_DIR / "scene.xml"):
        model = mujoco.MjModel.from_xml_path(str(xml_path))

        # 1) One Link per arm joint: fixed offset from the parent body + joint axis.
        self.links: list[Link] = []
        prev_body = None
        for jname in ARM_JOINTS:
            jid = model.joint(jname).id
            bid = model.jnt_bodyid[jid]
            assert np.allclose(model.jnt_pos[jid], 0.0), f"{jname}: joint not at body origin"
            if prev_body is not None:
                assert model.body_parentid[bid] == prev_body, f"{jname}: chain is not serial"
            T = make_transform(quat_to_rot(model.body_quat[bid]), model.body_pos[bid])
            self.links.append(Link(model.body(bid).name, T, model.jnt_axis[jid].copy()))
            prev_body = bid

        # 2) Fixed transform world -> parent of the first moving body (walk up the tree).
        first_body = model.jnt_bodyid[model.joint(ARM_JOINTS[0]).id]
        self.T_world_root = np.eye(4)
        b = model.body_parentid[first_body]
        while b != 0:  # body 0 is the world
            T_b = make_transform(quat_to_rot(model.body_quat[b]), model.body_pos[b])
            self.T_world_root = T_b @ self.T_world_root
            b = model.body_parentid[b]

        # 3) Fixed transform last body (gripper) -> TCP site.
        sid = model.site(TCP_SITE).id
        assert model.site_bodyid[sid] == prev_body, "TCP site is not on the last link"
        self.T_last_tcp = make_transform(quat_to_rot(model.site_quat[sid]), model.site_pos[sid])

        self.joint_limits = np.array([model.jnt_range[model.joint(j).id] for j in ARM_JOINTS])

    @property
    def n_joints(self) -> int:
        return len(self.links)

    def link_frames(self, q: np.ndarray) -> list[np.ndarray]:
        """World pose (4x4) of each moving body; frames[i] holds joint i's axis."""
        q = np.asarray(q, dtype=float)
        assert q.shape == (self.n_joints,), f"expected {self.n_joints} joint angles"
        T = self.T_world_root.copy()
        frames = []
        for link, qi in zip(self.links, q):
            T_joint = make_transform(axis_angle_to_rot(link.axis, qi), np.zeros(3))
            T = T @ link.T_parent_body @ T_joint   # fixed offset first, then the joint motion
            frames.append(T)
        return frames

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """4x4 world pose of the TCP (T_world_tcp) for joint angles q (radians)."""
        return self.link_frames(q)[-1] @ self.T_last_tcp


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    kin = SO101Kinematics()
    for q_deg in ([0, 0, 0, 0, 0], [30, -20, 45, 10, 0], [-60, 40, -30, 60, 90]):
        T = kin.forward_kinematics(np.radians(q_deg))
        print(f"q = {q_deg} deg -> TCP position [m] = {T[:3, 3]}")