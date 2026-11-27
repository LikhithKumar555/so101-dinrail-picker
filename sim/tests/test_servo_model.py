"""STS3215 physics model: reproduces the datasheet and basic control physics."""
import numpy as np
import pytest

from so101_sim.feetech.servo_model import KGCM, ServoArray, ServoParams


@pytest.fixture
def p():
    return ServoParams.sts3215_7v4()


def test_parameters_reproduce_the_datasheet(p):
    v = 7.4
    assert p.k_t * v / p.r_25 == pytest.approx(19.5 * KGCM, rel=1e-6)             # stall torque
    w_rated = 5.445 * (1 - 5 / 19.5)
    i_rated = (5 * KGCM + p.tau_coulomb + p.b_viscous * w_rated) / p.k_t
    assert i_rated == pytest.approx(0.65, rel=1e-3)                                  # rated current
    w0 = (v - p.r_25 * (p.tau_coulomb + p.b_viscous * 52 * 2 * np.pi / 60) / p.k_t) / p.k_e
    assert w0 == pytest.approx(52 * 2 * np.pi / 60, rel=1e-3)                        # no-load speed
    assert 0.3 < p.efficiency < 0.9                                                  # plausible gearbox


def test_no_load_speed_in_simulation(p):
    s = ServoArray(1, p)
    s.torque_enable[:] = True
    s.goal_steps[:] = 4000                    # far away -> duty saturates at 1
    s.step(n_steps=400)
    assert s.omega[0] == pytest.approx(52 * 2 * np.pi / 60, rel=0.02)


def test_holds_below_stall_and_is_back_driven_above(p):
    hold = ServoArray(1, p)
    hold.torque_enable[:] = True
    hold.goal_steps[:] = 2548
    hold.step(tau_load=-1.5, n_steps=2000)
    pos, cur = [], []
    for _ in range(1000):                             # 0.5 s of holding
        hold.step(tau_load=-1.5)
        pos.append(hold.position_steps[0])
        cur.append(hold.current[0])
    assert max(pos) - min(pos) <= 2                   # holds (at most a 1-step dither)
    assert abs(np.mean(cur) - 1.5 / p.k_t) <= p.tau_coulomb / p.k_t + 0.05
    over = ServoArray(1, p)
    over.torque_enable[:] = True
    over.goal_steps[:] = 2548
    over.step(tau_load=-2.2, n_steps=2000)    # more than the 1.91 N*m stall torque
    assert over.omega[0] < -0.5               # the load wins and turns the motor backwards


def test_holding_error_under_gravity_matches_p_control(p):
    s = ServoArray(1, p, j_load=0.002)
    s.torque_enable[:] = True
    s.step(tau_load=-0.3, n_steps=4000)
    err = 2048 - s.position_steps[0]
    assert abs(err - s.steady_state_error_steps(-0.3)) <= 1.5   # quantisation + dead zone


def test_step_response_settles_with_little_overshoot(p):
    s = ServoArray(1, p, j_load=0.002)
    s.torque_enable[:] = True
    s.goal_steps[:] = 2248
    pos = []
    for _ in range(1200):
        s.step()
        pos.append(s.position_steps[0])
    pos = np.array(pos)
    assert abs(pos[-1] - 2248) <= 2
    assert pos.max() - 2248 < 0.1 * 200       # < 10 % overshoot


def test_torque_off_coasts_and_stall_heats(p):
    s = ServoArray(2, p)
    s.torque_enable[:] = [False, True]
    s.goal_steps[:] = 2548
    s.step(tau_load=-3.0, n_steps=20000)      # 10 s against an immovable load
    assert s.current[0] == 0.0                # no torque: no current, the load just turns it
    assert s.temperature[1] > s.temperature[0] + 20


def test_encoder_is_quantised_and_wraps(p):
    s = ServoArray(1, p)
    s.set_position(np.pi - 1e-4)              # just below half a turn
    assert s.position_steps[0] == 4095 or s.position_steps[0] == 0
    s.set_position(0.0)
    assert s.position_steps[0] == 2048
