"""Feetech STS serial bus protocol (STS3215): packets, checksums, control table.

The STS servos speak a Dynamixel-1.0-style protocol on a half-duplex TTL bus (one wire for
both directions, default 1 Mbps, 8N1):

  instruction (host -> servo):  0xFF 0xFF  ID  LEN  INSTR  PARAM_1 .. PARAM_N  CHECKSUM
  status      (servo -> host):  0xFF 0xFF  ID  LEN  ERROR  PARAM_1 .. PARAM_N  CHECKSUM

  LEN      = N + 2                       (the INSTR/ERROR byte + the checksum)
  CHECKSUM = ~(ID + LEN + INSTR/ERROR + sum(PARAM)) & 0xFF
  ID 0xFE  = broadcast: every servo executes, none answers (two answers would collide on the wire)

Multi-byte registers of the STS series are little-endian (low byte first; the older SCS series is
big-endian). Signed values do not use two's complement but SIGN-MAGNITUDE: one bit is the sign,
the rest is the magnitude (bit 15 for velocities, bit 11 for the homing offset, bit 10 for load).

The register addresses follow the STS3215 memory table as used by LeRobot's Feetech driver.
Units and some bit meanings are this project's reading of the documentation - check them against
the official Feetech memory table before driving a real arm (Phase 8).
"""
from dataclasses import dataclass

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE

# instructions
PING, READ, WRITE, REG_WRITE, ACTION, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x04, 0x05, 0x82, 0x83

# status (error) byte bits - set by the servo in every reply
ERR_VOLTAGE, ERR_ANGLE, ERR_OVERHEAT, ERR_OVERCURRENT, ERR_CHECKSUM, ERR_OVERLOAD, ERR_INSTRUCTION = (
    1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6)

STS3215_MODEL_NUMBER = 777
STEPS_PER_REV = 4096                      # 12-bit magnetic encoder on the output shaft
CURRENT_UNIT_A = 0.0065                   # Present_Current: 6.5 mA per unit

# name: (address, size in bytes). Addresses < 40 are EEPROM (kept over power cycles, write-locked
# while Lock = 1), addresses >= 40 are RAM.
CONTROL_TABLE = {
    "Firmware_Major_Version": (0, 1), "Firmware_Minor_Version": (1, 1), "Model_Number": (3, 2),
    "ID": (5, 1), "Baud_Rate": (6, 1), "Return_Delay_Time": (7, 1), "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2), "Max_Position_Limit": (11, 2), "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1), "Min_Voltage_Limit": (15, 1), "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1), "Unloading_Condition": (19, 1), "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1), "D_Coefficient": (22, 1), "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2), "CW_Dead_Zone": (26, 1), "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2), "Angular_Resolution": (30, 1), "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1), "Protective_Torque": (34, 1), "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1), "Velocity_closed_loop_P": (37, 1),
    "Over_Current_Protection_Time": (38, 1), "Velocity_closed_loop_I": (39, 1),
    "Torque_Enable": (40, 1), "Acceleration": (41, 1), "Goal_Position": (42, 2), "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2), "Torque_Limit": (48, 2), "Lock": (55, 1),
    "Present_Position": (56, 2), "Present_Velocity": (58, 2), "Present_Load": (60, 2),
    "Present_Voltage": (62, 1), "Present_Temperature": (63, 1), "Status": (65, 1), "Moving": (66, 1),
    "Present_Current": (69, 2),
}
EEPROM_END = 40
SIGN_BIT = {"Homing_Offset": 11, "Goal_Velocity": 15, "Present_Velocity": 15, "Present_Load": 10}
BAUD_RATES = {0: 1_000_000, 1: 500_000, 2: 250_000, 3: 128_000, 4: 115_200, 5: 76_800,
              6: 57_600, 7: 38_400}


def checksum(body: bytes) -> int:
    """~(sum of ID, LEN, INSTR/ERROR and params) & 0xFF - catches any single-byte error."""
    return (~sum(body)) & 0xFF


def make_packet(servo_id: int, code: int, params: bytes = b"") -> bytes:
    """Build an instruction (code = INSTR) or a status packet (code = ERROR byte)."""
    body = bytes([servo_id, len(params) + 2, code]) + bytes(params)
    return HEADER + body + bytes([checksum(body)])


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    magnitude = abs(int(value))
    if magnitude >= (1 << sign_bit):
        raise ValueError(f"|{value}| does not fit below sign bit {sign_bit}")
    return magnitude | ((1 << sign_bit) if value < 0 else 0)


def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    magnitude = raw & ((1 << sign_bit) - 1)
    return -magnitude if raw & (1 << sign_bit) else magnitude


def to_bytes(name: str, value: int) -> bytes:
    """Register value -> little-endian bytes (sign-magnitude for the signed registers)."""
    size = CONTROL_TABLE[name][1]
    raw = encode_sign_magnitude(value, SIGN_BIT[name]) if name in SIGN_BIT else int(value)
    return raw.to_bytes(size, "little")


def from_bytes(name: str, data: bytes) -> int:
    raw = int.from_bytes(data, "little")
    return decode_sign_magnitude(raw, SIGN_BIT[name]) if name in SIGN_BIT else raw


# ------------------------------------------------------------------ instruction builders
def ping(servo_id: int) -> bytes:
    return make_packet(servo_id, PING)


def read(servo_id: int, name: str) -> bytes:
    addr, size = CONTROL_TABLE[name]
    return make_packet(servo_id, READ, bytes([addr, size]))


def write(servo_id: int, name: str, value: int) -> bytes:
    addr, _ = CONTROL_TABLE[name]
    return make_packet(servo_id, WRITE, bytes([addr]) + to_bytes(name, value))


def sync_write(name: str, values: dict) -> bytes:
    """One packet sets the same register on many servos: [addr, size, id1, data1.., id2, ..]."""
    addr, size = CONTROL_TABLE[name]
    params = bytes([addr, size])
    for sid, v in values.items():
        params += bytes([sid]) + to_bytes(name, v)
    return make_packet(BROADCAST_ID, SYNC_WRITE, params)


def sync_read(name: str, ids) -> bytes:
    """One request, then every listed servo answers in turn: [addr, size, id1, id2, ..]."""
    addr, size = CONTROL_TABLE[name]
    return make_packet(BROADCAST_ID, SYNC_READ, bytes([addr, size]) + bytes(ids))


# ------------------------------------------------------------------ parsing
@dataclass
class Packet:
    servo_id: int
    code: int              # instruction (host -> servo) or error byte (servo -> host)
    params: bytes


class PacketParser:
    """Byte-stream parser: feed() any chunk of bytes, get complete, checksum-valid packets.

    A serial line delivers bytes, not packets: a read can return half a packet or two packets,
    and noise can insert garbage. The parser searches for the 0xFF 0xFF header, waits until LEN
    bytes have arrived, and drops packets with a wrong checksum (counted in bad_checksums), then
    resynchronises on the next header.
    """

    def __init__(self):
        self.buf = bytearray()
        self.bad_checksums = 0

    def feed(self, data: bytes) -> list:
        self.buf += data
        out = []
        while True:
            start = self.buf.find(HEADER)
            if start < 0:
                self.buf = self.buf[-1:] if self.buf[-1:] == b"\xff" else bytearray()
                return out
            del self.buf[:start]
            if len(self.buf) < 4:
                return out
            if self.buf[2] == 0xFF:          # 0xFF 0xFF 0xFF ...: header shifted by one byte
                del self.buf[0]
                continue
            length = self.buf[3]
            total = 4 + length
            if length < 2:
                del self.buf[:2]
                continue
            if len(self.buf) < total:
                return out
            body = bytes(self.buf[2:total - 1])
            if checksum(body) == self.buf[total - 1]:
                out.append(Packet(body[0], body[2], bytes(body[3:])))
                del self.buf[:total]
            else:
                self.bad_checksums += 1
                del self.buf[:2]             # skip this header, look for the next one


def transfer_time_s(n_bytes: int, baud: int) -> float:
    """Time on the wire: every byte is 10 bits (start + 8 data + stop) at 8N1."""
    return n_bytes * 10 / baud
