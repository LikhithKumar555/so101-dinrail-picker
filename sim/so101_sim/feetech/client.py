"""Minimal Feetech bus client over a serial device - the Python reference for the C++ driver.

Only the standard library (termios) and the protocol module: no pyserial. It shows exactly what
a driver has to do on a half-duplex bus:
  1. open the port in raw mode, set the baud rate
  2. write one request, then read until the expected reply(s) arrived OR a timeout expired
  3. never wait for a reply to a broadcast (there is none)
  4. treat a missing or broken reply as an error the caller must handle (retry, stop the arm)
"""
import os
import select
import termios
import time
import tty

from so101_sim.feetech import protocol as P

BAUD = {1_000_000: termios.B1000000, 500_000: termios.B500000, 115_200: termios.B115200}


class BusTimeout(Exception):
    pass


class FeetechClient:
    def __init__(self, device: str, baud: int = 1_000_000, timeout_s: float = 0.02):
        self.fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[4] = attrs[5] = BAUD[baud]                 # input and output speed
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)      # drop stale bytes from earlier sessions
        self.timeout_s = timeout_s
        self.parser = P.PacketParser()
        self.timeouts = 0

    def close(self):
        os.close(self.fd)

    # ---------------------------------------------------------------- transport
    def _transact(self, request: bytes, n_replies: int):
        """Send one request, collect n_replies status packets (or raise BusTimeout)."""
        os.write(self.fd, request)
        replies = []
        deadline = time.monotonic() + self.timeout_s
        while len(replies) < n_replies:
            left = deadline - time.monotonic()
            if left <= 0:
                self.timeouts += 1
                raise BusTimeout(f"{len(replies)}/{n_replies} replies within {self.timeout_s * 1e3:.0f} ms")
            ready, _, _ = select.select([self.fd], [], [], left)
            if ready:
                replies += self.parser.feed(os.read(self.fd, 1024))
        return replies

    # ---------------------------------------------------------------- instructions
    def ping(self, sid: int) -> int:
        (r,) = self._transact(P.ping(sid), 1)
        return r.code

    def read(self, sid: int, name: str) -> int:
        (r,) = self._transact(P.read(sid, name), 1)
        return P.from_bytes(name, r.params)

    def write(self, sid: int, name: str, value: int) -> int:
        (r,) = self._transact(P.write(sid, name, value), 1)
        return r.code

    def sync_write(self, name: str, values: dict):
        os.write(self.fd, P.sync_write(name, values))      # broadcast: no reply to wait for

    def sync_read(self, name: str, ids) -> dict:
        ids = list(ids)
        replies = self._transact(P.sync_read(name, ids), len(ids))
        return {r.servo_id: P.from_bytes(name, r.params) for r in replies}
