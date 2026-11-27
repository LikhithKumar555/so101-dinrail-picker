"""Emulated Feetech STS3215 bus: answers real protocol bytes, backed by the servo physics model.

    host bytes --> PacketParser --> per-servo register memory  <--> ServoArray physics
              <-- status packets <--

The emulator is what a driver talks to before real hardware exists. It behaves like the real bus
in the ways a driver must handle:
  * register memory with EEPROM write lock, ID change, position limits and homing offset
  * broadcast writes get no answer; SYNC_READ makes every listed servo answer in turn
  * wrong checksums and unknown IDs get NO answer (the driver must time out, not hang)
  * the status byte reports overheating / overload / voltage errors; protections act like the
    servo's own (overheat -> torque off, sustained overload -> torque limited)
  * bus_time_s(): how long a transaction occupies the wire at the configured baud rate
"""
import numpy as np

from so101_sim.feetech import protocol as P
from so101_sim.feetech.protocol import CONTROL_TABLE, Packet, PacketParser
from so101_sim.feetech.servo_model import ServoArray, ServoParams

DEFAULTS = {
    "Firmware_Major_Version": 3, "Firmware_Minor_Version": 10, "Model_Number": P.STS3215_MODEL_NUMBER,
    "Baud_Rate": 0, "Return_Delay_Time": 0, "Response_Status_Level": 1,
    "Min_Position_Limit": 0, "Max_Position_Limit": 4095, "Max_Temperature_Limit": 70,
    "Max_Voltage_Limit": 80, "Min_Voltage_Limit": 40, "Max_Torque_Limit": 1000,
    "P_Coefficient": 32, "D_Coefficient": 32, "I_Coefficient": 0, "CW_Dead_Zone": 1, "CCW_Dead_Zone": 1,
    "Protection_Current": 500, "Homing_Offset": 0, "Operating_Mode": 0,
    "Protective_Torque": 20, "Protection_Time": 200, "Overload_Torque": 80,
    "Torque_Enable": 0, "Torque_Limit": 1000, "Goal_Position": 2048, "Lock": 1,
}


class FeetechBusEmulator:
    def __init__(self, ids=(1, 2, 3, 4, 5, 6), params: ServoParams | None = None, j_load=0.0,
                 supply_voltage: float = 7.4, dt: float = 5e-4):
        self.ids = list(ids)
        self.model = ServoArray(len(self.ids), params, j_load=j_load, dt=dt)
        self.model.supply_voltage = supply_voltage
        self.mem = {}
        for k, sid in enumerate(self.ids):
            m = bytearray(256)
            for name, value in DEFAULTS.items():
                self._put(m, name, value)
            self._put(m, "ID", sid)
            self.mem[sid] = m
        self.parser = PacketParser()
        self.pending = {}                    # REG_WRITE data, applied by ACTION
        self.overload_time = np.zeros(len(self.ids))
        self.time = 0.0
        self._sync_registers_to_model()

    # ---------------------------------------------------------------- register helpers
    @staticmethod
    def _put(m: bytearray, name: str, value: int):
        addr, size = CONTROL_TABLE[name]
        m[addr:addr + size] = P.to_bytes(name, value)

    @staticmethod
    def _get(m: bytearray, name: str) -> int:
        addr, size = CONTROL_TABLE[name]
        return P.from_bytes(name, bytes(m[addr:addr + size]))

    def register(self, sid: int, name: str) -> int:
        self._update_present(sid)
        return self._get(self.mem[sid], name)

    def _index(self, sid: int) -> int:
        return self.ids.index(sid)

    def _sync_registers_to_model(self):
        md = self.model
        for k, sid in enumerate(self.ids):
            m = self.mem[sid]
            lo, hi = self._get(m, "Min_Position_Limit"), self._get(m, "Max_Position_Limit")
            goal = int(np.clip(self._get(m, "Goal_Position"), lo, hi))
            md.goal_steps[k] = (goal + self._get(m, "Homing_Offset")) % P.STEPS_PER_REV
            md.torque_enable[k] = bool(self._get(m, "Torque_Enable"))
            limit = min(self._get(m, "Torque_Limit"), self._get(m, "Max_Torque_Limit")) / 1000.0
            if self.overload_time[k] < 0:                      # overload protection active
                limit = min(limit, self._get(m, "Protective_Torque") / 100.0)
            md.torque_limit[k] = limit
            md.kp[k] = self._get(m, "P_Coefficient")
            md.kd[k] = self._get(m, "D_Coefficient")
            md.dead_zone[k] = max(self._get(m, "CW_Dead_Zone"), self._get(m, "CCW_Dead_Zone"))

    def _update_present(self, sid: int):
        k, md, m = self._index(sid), self.model, self.mem[sid]
        raw = int(md.position_steps[k])
        self._put(m, "Present_Position", (raw - self._get(m, "Homing_Offset")) % P.STEPS_PER_REV)
        vel = int(np.clip(md.velocity_steps[k], -32767, 32767))
        self._put(m, "Present_Velocity", vel)
        self._put(m, "Present_Load", int(np.clip(round(md.duty[k] * 1000), -1000, 1000)))
        self._put(m, "Present_Voltage", int(round(md.supply_voltage * 10)))
        self._put(m, "Present_Temperature", int(np.clip(round(md.temperature[k]), 0, 255)))
        self._put(m, "Present_Current", int(min(abs(md.current[k]) / P.CURRENT_UNIT_A, 65535)))
        self._put(m, "Moving", int(abs(vel) > 8))
        self._put(m, "Status", self._error_byte(sid))

    def _error_byte(self, sid: int) -> int:
        k, md, m = self._index(sid), self.model, self.mem[sid]
        err = 0
        v10 = md.supply_voltage * 10
        if not self._get(m, "Min_Voltage_Limit") <= v10 <= self._get(m, "Max_Voltage_Limit"):
            err |= P.ERR_VOLTAGE
        if md.temperature[k] > self._get(m, "Max_Temperature_Limit"):
            err |= P.ERR_OVERHEAT
        if self.overload_time[k] < 0:
            err |= P.ERR_OVERLOAD
        return err

    # ---------------------------------------------------------------- physics clock
    def advance(self, seconds: float, tau_load=0.0):
        """Let time pass on the servos (physics + protections)."""
        md = self.model
        n = max(1, int(round(seconds / md.dt)))
        for _ in range(n):
            md.step(tau_load)
            self.time += md.dt
            for k, sid in enumerate(self.ids):
                m = self.mem[sid]
                if md.temperature[k] > self._get(m, "Max_Temperature_Limit") and md.torque_enable[k]:
                    self._put(m, "Torque_Enable", 0)           # overheat: the servo unloads itself
                    md.torque_enable[k] = False
                over = abs(md.duty[k]) * 100 > self._get(m, "Overload_Torque")
                if self.overload_time[k] >= 0:
                    self.overload_time[k] = self.overload_time[k] + md.dt if over else 0.0
                    if self.overload_time[k] > self._get(m, "Protection_Time") * 0.01:
                        self.overload_time[k] = -1.0          # protection latched
                        self._sync_registers_to_model()

    # ---------------------------------------------------------------- protocol
    def handle_bytes(self, data: bytes) -> bytes:
        """Raw bytes from the host -> raw reply bytes (possibly empty)."""
        out = b""
        for pkt in self.parser.feed(data):
            out += self.handle(pkt)
        return out

    def _status(self, sid: int, params: bytes = b"", extra_error: int = 0) -> bytes:
        return P.make_packet(sid, self._error_byte(sid) | extra_error, params)

    def _write_mem(self, sid: int, addr: int, data: bytes):
        """Write bytes into one servo's memory -> (error bit, the servo's ID afterwards)."""
        m = self.mem[sid]
        if addr < P.EEPROM_END and self._get(m, "Lock"):
            return P.ERR_INSTRUCTION, sid                      # EEPROM is write-locked
        m[addr:addr + len(data)] = data
        a_tq = CONTROL_TABLE["Torque_Enable"][0]
        if addr <= a_tq < addr + len(data) and self._get(m, "Torque_Enable"):
            k = self._index(sid)
            if self.overload_time[k] < 0:                      # re-enabling clears a latched overload
                self.overload_time[k] = 0.0
        new_id = self._get(m, "ID")
        if new_id != sid:                                      # ID changed (EEPROM was unlocked)
            self.ids[self._index(sid)] = new_id
            self.mem[new_id] = self.mem.pop(sid)
        self._sync_registers_to_model()
        return 0, new_id

    def handle(self, pkt: Packet) -> bytes:
        sid, ins, prm = pkt.servo_id, pkt.code, pkt.params
        broadcast = sid == P.BROADCAST_ID
        if ins == P.SYNC_WRITE:
            addr, size = prm[0], prm[1]
            for j in range(2, len(prm), size + 1):
                tid = prm[j]
                if tid in self.mem:
                    self._write_mem(tid, addr, prm[j + 1:j + 1 + size])
            return b""
        if ins == P.SYNC_READ:
            addr, size = prm[0], prm[1]
            out = b""
            for tid in prm[2:]:                                # each servo answers in its turn
                if tid in self.mem:
                    self._update_present(tid)
                    out += self._status(tid, bytes(self.mem[tid][addr:addr + size]))
            return out
        if ins == P.ACTION:
            for tid, (addr, data) in list(self.pending.items()):
                if tid in self.mem:
                    self._write_mem(tid, addr, data)
            self.pending.clear()
            return b""
        targets = self.ids if broadcast else ([sid] if sid in self.mem else [])
        if not targets:
            return b""                                         # nobody with that ID: silence
        out = b""
        for tid in targets:
            if ins == P.PING:
                reply = self._status(tid)
            elif ins == P.READ:
                addr, size = prm[0], prm[1]
                self._update_present(tid)
                reply = self._status(tid, bytes(self.mem[tid][addr:addr + size]))
            elif ins == P.WRITE:
                err, tid = self._write_mem(tid, prm[0], prm[1:])  # [ASSUMED] reply from the new ID
                reply = self._status(tid, extra_error=err)
            elif ins == P.REG_WRITE:
                self.pending[tid] = (prm[0], bytes(prm[1:]))
                reply = self._status(tid)
            else:
                reply = self._status(tid, extra_error=P.ERR_INSTRUCTION)
            if not broadcast:
                out += reply
        return out

    # ---------------------------------------------------------------- timing
    def bus_time_s(self, request: bytes, reply: bytes, sid: int | None = None) -> float:
        """Wire time of one transaction: request + return delay + reply."""
        sid = self.ids[0] if sid is None else sid
        m = self.mem[sid]
        baud = P.BAUD_RATES[self._get(m, "Baud_Rate")]
        delay = self._get(m, "Return_Delay_Time") * 2e-6 if reply else 0.0
        return P.transfer_time_s(len(request) + len(reply), baud) + delay
