import mujoco
import numpy as np
import pytest

from so101_sim.kinematics import (ARM_JOINTS, MODEL_DIR, TCP_SITE, SO101Kinematics,
                                  axis_angle_to_rot, make_transform, quat_to_rot, skew)

rng = np.random.default_rng(0)


def random_unit(n):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


def test_quat_to_rot_matches_mujoco():
    for _ in range(100):
        quat = random_unit(4)
        expected = np.zeros(9)
        mujoco.mju_quat2Mat(expected, quat)
        np.testing.assert_allclose(quat_to_rot(quat), expected.reshape(3, 3), atol=1e-12)


def test_skew_is_cross_product():
    for _ in range(100):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-12)


def test_axis_angle_to_rot():
    for _ in range(100):
        axis, theta = random_unit(3), rng.uniform(-np.pi, np.pi)
        R = axis_angle_to_rot(axis, theta)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)   # orthonormal
        assert np.isclose(np.linalg.det(R), 1.0)                     # proper rotation
        np.testing.assert_allclose(R @ axis, axis, atol=1e-12)       # axis unchanged
        quat, expected = np.zeros(4), np.zeros(9)
        mujoco.mju_axisAngle2Quat(quat, axis, theta)
        mujoco.mju_quat2Mat(expected, quat)
        np.testing.assert_allclose(R, expected.reshape(3, 3), atol=1e-12)


def test_make_transform():
    R, p = axis_angle_to_rot(np.array([0.0, 0.0, 1.0]), 0.3), np.array([1.0, 2.0, 3.0])
    T = make_transform(R, p)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], R)
    np.testing.assert_allclose(T[:3, 3], p)
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    return model, mujoco.MjData(model)


def test_fk_matches_mujoco(mj):
    model, data = mj
    kin = SO101Kinematics()
    qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    sid = model.site(TCP_SITE).id
    for _ in range(500):
        q = rng.uniform(kin.joint_limits[:, 0], kin.joint_limits[:, 1])
        data.qpos[:] = 0.0
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        T = kin.forward_kinematics(q)
        np.testing.assert_allclose(T[:3, 3], data.site_xpos[sid], atol=1e-9)
        np.testing.assert_allclose(T[:3, :3], data.site_xmat[sid].reshape(3, 3), atol=1e-9)