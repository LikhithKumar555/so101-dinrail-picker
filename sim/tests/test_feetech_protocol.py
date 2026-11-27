"""Feetech protocol: checksums, packet layout, sign-magnitude, stream parsing."""
import pytest

from so101_sim.feetech import protocol as P


def test_ping_packet_matches_the_documented_bytes():
    # PING to ID 1: FF FF 01 02 01 FB  (checksum = ~(1 + 2 + 1) & 0xFF = 0xFB)
    assert P.ping(1) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_write_goal_position_is_little_endian():
    pkt = P.write(1, "Goal_Position", 2048)            # 2048 = 0x0800 -> low byte first: 00 08
    assert pkt[:5] == bytes([0xFF, 0xFF, 1, 5, P.WRITE])
    assert pkt[5:8] == bytes([42, 0x00, 0x08])
    assert pkt[-1] == P.checksum(pkt[2:-1])


def test_sign_magnitude_is_not_twos_complement():
    assert P.to_bytes("Present_Velocity", -5) == bytes([0x05, 0x80])    # bit 15 set, magnitude 5
    assert P.from_bytes("Present_Velocity", bytes([0x05, 0x80])) == -5
    assert P.from_bytes("Present_Load", (1024 + 300).to_bytes(2, "little")) == -300   # bit 10
    with pytest.raises(ValueError):
        P.to_bytes("Homing_Offset", 3000)                # magnitude must stay below bit 11


def test_sync_write_layout():
    pkt = P.sync_write("Goal_Position", {1: 100, 2: 200})
    assert pkt[2] == P.BROADCAST_ID and pkt[4] == P.SYNC_WRITE
    assert pkt[5:7] == bytes([42, 2])                    # address, bytes per servo
    assert pkt[7:10] == bytes([1, 100, 0]) and pkt[10:13] == bytes([2, 200, 0])
    assert pkt[3] == len(pkt) - 4                        # LEN counts from INSTR to the checksum


def test_parser_handles_split_glued_and_corrupted_streams():
    a, b = P.ping(1), P.read(2, "Present_Position")
    stream = b"\x00\x13" + a + b
    parser = P.PacketParser()
    got = []
    for i in range(0, len(stream), 3):                   # arrives in 3-byte chunks
        got += parser.feed(stream[i:i + 3])
    assert [(g.servo_id, g.code) for g in got] == [(1, P.PING), (2, P.READ)]

    bad = bytearray(P.ping(3))
    bad[-1] ^= 0xFF                                      # corrupt the checksum
    got = P.PacketParser().feed(bytes(bad) + P.ping(4))
    assert [g.servo_id for g in got] == [4]              # bad packet dropped, next one found


def test_wire_time_at_1_mbps():
    # 8N1: 10 bits per byte -> 10 us per byte at 1 Mbps
    assert P.transfer_time_s(10, 1_000_000) == pytest.approx(100e-6)
