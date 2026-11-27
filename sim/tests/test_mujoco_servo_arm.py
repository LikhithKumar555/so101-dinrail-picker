"""Emulated servos driving the MuJoCo SO-101: holding under gravity, moving, releasing."""
import numpy as np
import pytest

from so101_sim.feetech import protocol as P
from so101_sim.feetech.mujoco_arm import ServoArm
from so101_sim.feetech.servo_model import RAD_PER_STEP
from so101_sim.scene import ready_q


@pytest.fixture
def arm():
    a = ServoArm(q0=ready_q())
    a.bus.handle_bytes(P.sync_write("Torque_Enable", {i: 1 for i in range(1, 7)}))
    return a


def test_servo_parameters_are_in_the_model(arm):
    p = arm.bus.model.p
    assert np.allclose(arm.m.dof_armature[arm.dadr], p.j_reflected)       # J_rotor * N^2
    assert np.allclose(arm.m.dof_frictionloss[arm.dadr], p.tau_coulomb)
    assert np.all(arm.m.actuator_biastype[arm.act] == 0)                   # plain torque motors


def test_sag_under_gravity_matches_p_control(arm):
    q0 = arm.q
    arm.advance(1.0)
    tau_g = arm.gravity_torque()
    md = arm.bus.model
    for k in (1, 2):                                    # shoulder_lift, elbow_flex carry the arm
        sag_steps = abs(arm.q[k] - q0[k]) / RAD_PER_STEP
        expected = md.steady_state_error_steps(tau_g[k])
        assert sag_steps == pytest.approx(expected, abs=2.0)
        assert np.sign(arm.q[k] - q0[k]) == np.sign(tau_g[k])   # it sags in the direction of gravity


def test_goal_positions_are_reached(arm):
    goals = {i: int(arm.bus.model.goal_steps[i - 1]) for i in range(1, 7)}
    goals[1] += 300
    goals[3] -= 200
    arm.bus.handle_bytes(P.sync_write("Goal_Position", goals))
    arm.advance(1.0)
    replies = P.PacketParser().feed(arm.bus.handle_bytes(P.sync_read("Present_Position", range(1, 7))))
    present = {r.servo_id: P.from_bytes("Present_Position", r.params) for r in replies}
    for i in (1, 3):
        assert abs(present[i] - goals[i]) <= 12        # within the gravity sag


def test_torque_off_lets_the_arm_fall(arm):
    arm.advance(0.2)
    lift = arm.q[1]
    arm.bus.handle_bytes(P.sync_write("Torque_Enable", {i: 0 for i in range(1, 7)}))
    arm.advance(0.5)
    assert abs(arm.q[1] - lift) > 0.2                  # no torque -> gravity wins
    assert np.all(arm.bus.model.current == 0.0)
