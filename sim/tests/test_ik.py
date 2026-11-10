import mujoco
import numpy as np
import pytest

from so101_sim.ik import (geometric_jacobian, orientation_error, pose_error, rot_to_quat,
                          solve_ik, top_down_pose)
from so101_sim.kinematics import (ARM_JOINTS, MODEL_DIR, TCP_SITE, SO101Kinematics,
                                  axis_angle_to_rot, quat_to_rot)

rng = np.random.default_rng(1)


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    return model, mujoco.MjData(model)


def random_q(kin):
    return rng.uniform(kin.joint_limits[:, 0], kin.joint_limits[:, 1])


def test_rot_to_quat_roundtrip():
    for _ in range(200):
        axis = rng.normal(size=3)
        R = axis_angle_to_rot(axis, rng.uniform(-np.pi, np.pi))
        np.testing.assert_allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-10)


def test_orientation_error_recovers_rotation():
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, 3.0)
        R = axis_angle_to_rot(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        R_target = axis_angle_to_rot(axis, angle) @ R
        np.testing.assert_allclose(orientation_error(R_target, R), axis * angle, atol=1e-9)


def test_jacobian_matches_mujoco(kin, mj):
    model, data = mj
    qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    dofadr = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
    sid = model.site(TCP_SITE).id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(200):
        q = random_q(kin)
        data.qpos[:] = 0.0
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = geometric_jacobian(kin, q)
        np.testing.assert_allclose(J[:3], jacp[:, dofadr], atol=1e-9)
        np.testing.assert_allclose(J[3:], jacr[:, dofadr], atol=1e-9)


def test_jacobian_matches_finite_differences(kin):
    eps = 1e-6
    for _ in range(50):
        q = random_q(kin)
        T0 = kin.forward_kinematics(q)
        J = geometric_jacobian(kin, q)
        for i in range(kin.n_joints):
            dq = np.zeros(kin.n_joints)
            dq[i] = eps
            twist = pose_error(kin.forward_kinematics(q + dq), T0) / eps
            np.testing.assert_allclose(J[:, i], twist, atol=1e-5)


def test_ik_reaches_random_reachable_poses(kin):
    n, ok = 100, 0
    for _ in range(n):
        T_target = kin.forward_kinematics(random_q(kin))
        res = solve_ik(kin, T_target)
        ok += res.success
        if res.success:
            T = kin.forward_kinematics(res.q)
            assert np.linalg.norm(T[:3, 3] - T_target[:3, 3]) < 1e-4
            assert np.all(res.q >= kin.joint_limits[:, 0] - 1e-9)
            assert np.all(res.q <= kin.joint_limits[:, 1] + 1e-9)
    assert ok / n >= 0.90, f"IK success rate only {ok}/{n}"


def test_ik_top_down_grasps_on_table(kin):
    for x, y, yaw in [(0.20, 0.00, 0.0), (0.25, 0.10, 0.5), (0.18, -0.12, -0.8), (0.30, 0.05, 1.2)]:
        res = solve_ik(kin, top_down_pose([x, y, 0.02], yaw))
        assert res.success, f"no top-down solution at {(x, y, yaw)}: {res}"
        approach = kin.forward_kinematics(res.q)[:3, 0]
        assert approach[2] < -0.999            # gripper points straight down
