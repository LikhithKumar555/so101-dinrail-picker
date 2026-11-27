"""Emulated Feetech bus: answers like real servos, byte for byte."""
import numpy as np
import pytest

from so101_sim.feetech import protocol as P
from so101_sim.feetech.emulator import FeetechBusEmulator


def ask(bus, request: bytes):
    """Send request bytes, parse the reply bytes into packets."""
    return P.PacketParser().feed(bus.handle_bytes(request))


@pytest.fixture
def bus():
    return FeetechBusEmulator(ids=(1, 2, 3))


def test_ping_and_model_number(bus):
    (r,) = ask(bus, P.ping(2))
    assert r.servo_id == 2 and r.code == 0
    (r,) = ask(bus, P.read(1, "Model_Number"))
    assert P.from_bytes("Model_Number", r.params) == P.STS3215_MODEL_NUMBER


def test_silence_for_unknown_id_bad_checksum_and_broadcast(bus):
    assert ask(bus, P.ping(9)) == []
    bad = bytearray(P.ping(1))
    bad[-1] ^= 1
    assert bus.handle_bytes(bytes(bad)) == b""
    assert bus.handle_bytes(P.write(P.BROADCAST_ID, "Torque_Enable", 1)) == b""
    assert all(bus.register(i, "Torque_Enable") == 1 for i in (1, 2, 3))


def test_goal_position_moves_the_servo(bus):
    ask(bus, P.write(1, "Torque_Enable", 1))
    ask(bus, P.write(1, "Goal_Position", 2548))
    bus.advance(0.5)
    (r,) = ask(bus, P.read(1, "Present_Position"))
    assert abs(P.from_bytes("Present_Position", r.params) - 2548) <= 2
    assert bus.register(2, "Present_Position") == 2048        # the others did not move


def test_sync_write_and_sync_read(bus):
    bus.handle_bytes(P.sync_write("Torque_Enable", {1: 1, 2: 1, 3: 1}))
    bus.handle_bytes(P.sync_write("Goal_Position", {1: 1848, 2: 2248, 3: 2448}))
    bus.advance(0.5)
    replies = ask(bus, P.sync_read("Present_Position", [1, 2, 3]))
    assert [r.servo_id for r in replies] == [1, 2, 3]          # answered in the listed order
    pos = [P.from_bytes("Present_Position", r.params) for r in replies]
    assert np.allclose(pos, [1848, 2248, 2448], atol=2)


def test_eeprom_lock_and_id_change(bus):
    (r,) = ask(bus, P.write(3, "ID", 7))
    assert r.code & P.ERR_INSTRUCTION and 3 in bus.ids       # locked: refused
    ask(bus, P.write(3, "Lock", 0))
    (r,) = ask(bus, P.write(3, "ID", 7))
    assert r.code == 0 and 7 in bus.ids and 3 not in bus.ids
    assert ask(bus, P.ping(7))[0].servo_id == 7


def test_position_limits_clamp_the_goal(bus):
    ask(bus, P.write(1, "Lock", 0))
    ask(bus, P.write(1, "Max_Position_Limit", 2300))
    ask(bus, P.write(1, "Torque_Enable", 1))
    ask(bus, P.write(1, "Goal_Position", 3000))
    bus.advance(0.6)
    assert abs(bus.register(1, "Present_Position") - 2300) <= 2


def test_overload_protection_limits_torque_and_reports_it(bus):
    ask(bus, P.write(1, "Torque_Enable", 1))
    ask(bus, P.write(1, "Goal_Position", 2548))
    bus.advance(3.0, tau_load=np.array([-1.7, 0.0, 0.0]))   # holds 1.7 N*m: about 90 % duty
    (r,) = ask(bus, P.ping(1))
    assert r.code & P.ERR_OVERLOAD
    assert bus.model.torque_limit[0] == pytest.approx(0.20)  # Protective_Torque = 20 %


def test_transaction_time_on_the_wire(bus):
    req = P.sync_read("Present_Position", [1, 2, 3])
    rep = bus.handle_bytes(req)
    # 11 request bytes + 3 x 8 reply bytes = 35 bytes -> 350 us at 1 Mbps
    assert bus.bus_time_s(req, rep) == pytest.approx(35 * 10e-6)
