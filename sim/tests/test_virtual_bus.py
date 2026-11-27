"""The emulated bus behind a real pseudo-terminal, used through a termios client."""
import pytest

from so101_sim.feetech import protocol as P
from so101_sim.feetech.client import BusTimeout, FeetechClient
from so101_sim.feetech.emulator import FeetechBusEmulator
from so101_sim.feetech.virtual_bus import VirtualSerialBus


@pytest.fixture
def bus(tmp_path):
    vb = VirtualSerialBus(FeetechBusEmulator(), link=str(tmp_path / "so101_bus")).start()
    client = FeetechClient(vb.device, timeout_s=0.02)
    yield vb, client
    client.close()
    vb.stop()


def test_ping_read_and_sync_read_over_the_pty(bus):
    vb, c = bus
    assert [c.ping(i) for i in range(1, 7)] == [0] * 6
    assert c.read(3, "Model_Number") == P.STS3215_MODEL_NUMBER
    assert c.sync_read("Present_Position", range(1, 7)) == {i: 2048 for i in range(1, 7)}


def test_unknown_id_times_out(bus):
    vb, c = bus
    with pytest.raises(BusTimeout):
        c.ping(42)
    assert c.ping(1) == 0                     # the bus is fine afterwards


def test_raw_mode_keeps_control_characters(bus):
    """0x0D (carriage return) and 0x03 (Ctrl-C) would be changed or eaten by a terminal in
    its default 'cooked' mode. Goal_Position 3331 = 0x0D03 sends exactly those bytes."""
    vb, c = bus
    assert P.to_bytes("Goal_Position", 3331) == bytes([0x03, 0x0D])
    c.write(1, "Goal_Position", 3331)
    assert c.read(1, "Goal_Position") == 3331


def test_servos_move_in_real_time(bus):
    import time
    vb, c = bus
    c.sync_write("Torque_Enable", {1: 1})
    c.sync_write("Goal_Position", {1: 2348})
    time.sleep(0.6)                           # the physics runs against the wall clock
    assert abs(c.read(1, "Present_Position") - 2348) <= 3


def test_dropped_replies_are_timeouts(tmp_path):
    vb = VirtualSerialBus(FeetechBusEmulator(), link=str(tmp_path / "bad_bus"), drop_rate=1.0).start()
    c = FeetechClient(vb.device, timeout_s=0.01)
    try:
        with pytest.raises(BusTimeout):
            c.sync_read("Present_Position", [1, 2])
        assert vb.stats["dropped"] >= 1
    finally:
        c.close()
        vb.stop()
