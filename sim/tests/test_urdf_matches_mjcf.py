"""The ROS URDF (ros2_ws/src/so101_description) and the MuJoCo MJCF must describe the same robot.

Parses the URDF with plain XML, computes FK with our own transform helpers, and compares
with MuJoCo for random joint angles: gripper link and TCP must coincide.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_sim.kinematics import (ARM_JOINTS, MODEL_DIR, SO101Kinematics, axis_angle_to_rot,
                                  make_transform)

PKG = Path(__file__).resolve().parents[2] / "ros2_ws" / "src" / "so101_description" / "urdf"


def rpy_to_rot(r, p, y):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (fixed-axis roll, pitch, yaw)."""
    ez, ey, ex = np.eye(3)[2], np.eye(3)[1], np.eye(3)[0]
    return axis_angle_to_rot(ez, y) @ axis_angle_to_rot(ey, p) @ axis_angle_to_rot(ex, r)


def load_joints():
    joints = []
    for f in ("so101_arm.urdf", "so101.urdf.xacro"):
        joints += ET.parse(PKG / f).getroot().findall("joint")
    return {j.find("child").get("link"): j for j in joints}


def urdf_fk(by_child, link, q):
    T = np.eye(4)
    while link in by_child:
        j = by_child[link]
        o = j.find("origin")
        T_j = make_transform(rpy_to_rot(*map(float, o.get("rpy").split())),
                             np.array(o.get("xyz").split(), float))
        if j.get("type") == "revolute":
            axis = np.array(j.find("axis").get("xyz").split(), float)
            T_j = T_j @ make_transform(axis_angle_to_rot(axis, q[j.get("name")]), np.zeros(3))
        T = T_j @ T
        link = j.find("parent").get("link")
    return T


@pytest.mark.skipif(not (PKG / "so101_arm.urdf").exists(), reason="URDF not imported yet")
def test_urdf_and_mjcf_kinematics_agree():
    by_child = load_joints()
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    data = mujoco.MjData(model)
    kin = SO101Kinematics()
    qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    rng = np.random.default_rng(3)
    for _ in range(100):
        q = rng.uniform(kin.joint_limits[:, 0], kin.joint_limits[:, 1])
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        qd = dict(zip(ARM_JOINTS, q))
        T_mj = make_transform(data.body("gripper").xmat.reshape(3, 3), data.body("gripper").xpos)
        np.testing.assert_allclose(urdf_fk(by_child, "gripper_link", qd), T_mj, atol=1e-4)
        np.testing.assert_allclose(urdf_fk(by_child, "tcp_link", qd), kin.forward_kinematics(q),
                                   atol=1e-4)
