"""The emulated servo bus as a serial device: a pseudo-terminal (pty) that any program can open.

    your driver  --open("/tmp/so101_bus")-->  pty (kernel)  <--master fd--  VirtualSerialBus
                                                                             |-- FeetechBusEmulator
                                                                             '-- physics in real time

A pty is a pair of connected file descriptors made by the kernel: what is written to one end
can be read at the other. The "slave" end looks exactly like a serial port (/dev/pts/N, with
termios settings), so a driver cannot tell it from /dev/ttyACM0. Two details matter:
  * RAW mode: by default a terminal edits the byte stream (echo, CR -> LF, Ctrl-C ...). Binary
    protocols need raw mode, on both the emulator and the driver side, or bytes like 0x0D or 0x03
    get changed or eaten.
  * TIMING: the kernel delivers bytes instantly, so the server waits as long as the real wire
    would take (10 us per byte at 1 Mbps + the servo's return delay) before it answers. Optional
    extra latency and dropped replies let a driver's timeout handling be tested.

The physics runs against the wall clock: every loop pass first advances the servos (and the
MuJoCo arm) until simulated time has caught up with real time, then serves the bus.
"""
import os
import select
import threading
import time
import tty
from pathlib import Path

import numpy as np

from so101_sim.feetech import protocol as P


class VirtualSerialBus:
    def __init__(self, bus, link: str = "/tmp/so101_bus", extra_latency_s: float = 0.0,
                 drop_rate: float = 0.0, seed: int = 0, realtime: bool = True, on_tick=None):
        """bus: FeetechBusEmulator (servo physics) or ServoArm (MuJoCo-coupled, has .bus)."""
        self.arm = bus if hasattr(bus, "bus") else None
        self.bus = bus.bus if self.arm else bus
        self.master, self.slave = os.openpty()
        tty.setraw(self.master)
        tty.setraw(self.slave)                       # binary-safe: no echo, no CR/LF translation
        self.path = os.ttyname(self.slave)
        self.link = Path(link) if link else None
        if self.link:
            if self.link.is_symlink() or self.link.exists():
                self.link.unlink()
            self.link.symlink_to(self.path)
        self.extra_latency_s, self.drop_rate = extra_latency_s, drop_rate
        self.rng = np.random.default_rng(seed)
        self.realtime = realtime
        self.on_tick = on_tick                       # e.g. a viewer sync, called ~30 times a second
        self.stats = {"requests": 0, "replies": 0, "dropped": 0, "bytes_in": 0, "bytes_out": 0,
                      "lag_s": 0.0}
        self._stop = threading.Event()
        self._thread = None

    @property
    def device(self) -> str:
        return str(self.link) if self.link else self.path

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        self._thread = threading.Thread(target=self.serve, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass
        if self.link and self.link.is_symlink():
            self.link.unlink()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- main loop
    def _advance_to(self, t_sim: float, budget_s: float = 0.002):
        """Catch the physics up with real time, but never block the bus for more than budget_s:
        a request must be answered within the driver's timeout even if the physics lags."""
        dt = self.bus.model.dt
        t_end = time.monotonic() + budget_s
        while self.bus.time + dt <= t_sim and time.monotonic() < t_end:
            self.bus.advance(dt * min(5, max(1, int((t_sim - self.bus.time) / dt))))
        self.stats["lag_s"] = max(0.0, t_sim - self.bus.time)    # > 0: physics slower than real time

    def serve(self):
        t0 = time.monotonic()
        sim0 = self.bus.time
        next_tick = 0.0
        while not self._stop.is_set():
            now = time.monotonic() - t0
            if self.realtime:
                self._advance_to(sim0 + now)
            if self.on_tick and now >= next_tick:
                self.on_tick()
                next_tick = now + 1 / 30
            try:
                ready, _, _ = select.select([self.master], [], [], 0.0005)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(self.master, 4096)
            except OSError:
                return
            self.stats["bytes_in"] += len(data)
            for pkt in self.bus.parser.feed(data):
                self.stats["requests"] += 1
                request = P.make_packet(pkt.servo_id, pkt.code, pkt.params)
                reply = self.bus.handle(pkt)
                if not reply:
                    continue
                if self.drop_rate and self.rng.random() < self.drop_rate:
                    self.stats["dropped"] += 1           # the servo "did not hear" the request
                    continue
                wait = self.bus.bus_time_s(request, reply) + self.extra_latency_s
                t_end = time.monotonic() + wait
                while time.monotonic() < t_end:          # busy-wait: sleep() is too coarse for 100 us
                    pass
                os.write(self.master, reply)
                self.stats["replies"] += 1
                self.stats["bytes_out"] += len(reply)
