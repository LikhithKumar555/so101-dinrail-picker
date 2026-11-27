"""A control loop over the (virtual) Feetech bus - what a ros2_control driver does every cycle.

  terminal 1:  python sim/scripts/run_virtual_bus.py --viewer
  terminal 2:  python sim/scripts/bus_client_demo.py --seconds 6

1. ping all six servos, read the model number (777 = STS3215)
2. bus timing: six single READs vs one SYNC_READ for the same six positions
3. safe start: goal := present position BEFORE torque on (the servo keeps its old goal in RAM and
   would jump to it), then ramp slowly to the ready pose
4. control loop: SYNC_WRITE goals (a slow wave on shoulder and elbow) + SYNC_READ positions,
   as fast as possible or at a fixed --rate; reports the loop rate, latency, timeouts and
   the tracking error, and writes media/virtual_bus_loop.png
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from so101_sim.feetech.client import BusTimeout, FeetechClient
from so101_sim.feetech.servo_model import ServoArray
from so101_sim.scene import ready_q

ROOT = Path(__file__).resolve().parents[2]
IDS = [1, 2, 3, 4, 5, 6]


def main(a):
    c = FeetechClient(a.device, timeout_s=a.timeout_ms / 1000)
    models = {i: c.read(i, "Model_Number") for i in IDS}
    print("servos found:", {i: ("STS3215" if m == 777 else m) for i, m in models.items()})

    n = 50
    t = time.perf_counter()
    for _ in range(n):
        for i in IDS:
            c.read(i, "Present_Position")
    single = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for _ in range(n):
        c.sync_read("Present_Position", IDS)
    sync = (time.perf_counter() - t) / n
    print(f"read 6 positions: 6 x READ {single * 1e3:.2f} ms   vs   1 x SYNC_READ {sync * 1e3:.2f} ms")

    # safe start: hold where the arm is now, then move slowly to the ready pose
    here = c.sync_read("Present_Position", IDS)
    c.sync_write("Goal_Position", here)
    c.sync_write("Torque_Enable", {i: 1 for i in IDS})
    ready = dict(zip(IDS, ServoArray.to_steps(np.append(ready_q(), 0.0)).tolist()))
    t0 = time.monotonic()
    while (s := (time.monotonic() - t0) / a.ramp_s) < 1.0:
        w = 0.5 - 0.5 * np.cos(np.pi * s)                    # smooth start and stop
        c.sync_write("Goal_Position", {i: int(round(here[i] + w * (ready[i] - here[i]))) for i in IDS})
        time.sleep(0.01)
    c.sync_write("Goal_Position", ready)
    time.sleep(0.5)
    print(f"ramped from {here} to the ready pose {ready}")
    start = ready
    log_t, log_goal, log_pos, lat = [], [], [], []
    t0 = time.monotonic()
    period = 1.0 / a.rate if a.rate else 0.0
    next_t = 0.0
    while (now := time.monotonic() - t0) < a.seconds:
        wave = int(a.amplitude * np.sin(2 * np.pi * now / a.wave_period))
        goals = dict(start)
        goals[2] = start[2] + wave
        goals[3] = start[3] - wave
        c.sync_write("Goal_Position", goals)
        t1 = time.monotonic()
        try:
            pos = c.sync_read("Present_Position", IDS)
            lat.append(time.monotonic() - t1)
            log_t.append(now)
            log_goal.append(goals[2])
            log_pos.append(pos[2])
        except BusTimeout:
            pass
        if period:
            next_t += period
            while time.monotonic() - t0 < next_t:
                pass
    c.sync_write("Goal_Position", start)
    cycles = len(lat) + c.timeouts
    lat = np.array(lat) * 1e3
    err = np.array(log_goal) - np.array(log_pos)
    print(f"control loop: {cycles / a.seconds:.0f} Hz, SYNC_READ latency median {np.median(lat):.2f} ms, "
          f"p99 {np.percentile(lat, 99):.2f} ms, timeouts {c.timeouts}/{cycles}")
    print(f"shoulder_lift tracking error: rms {np.sqrt(np.mean(err ** 2)):.1f} steps "
          f"({np.degrees(np.sqrt(np.mean(err ** 2)) * 2 * np.pi / 4096):.2f} deg), max {np.abs(err).max()} steps")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(log_t, log_goal, label="goal (SYNC_WRITE)")
    ax[0].plot(log_t, log_pos, label="present (SYNC_READ)")
    ax[0].set_xlabel("time [s]"), ax[0].set_ylabel("shoulder_lift [steps]"), ax[0].legend(), ax[0].grid(alpha=0.3)
    ax[0].set_title("servo 2 following a wave over the virtual bus")
    ax[1].hist(lat, bins=50)
    ax[1].set_xlabel("SYNC_READ round trip [ms]"), ax[1].set_ylabel("count"), ax[1].grid(alpha=0.3)
    ax[1].set_title(f"{cycles / a.seconds:.0f} Hz loop, {c.timeouts} timeouts")
    fig.tight_layout()
    out = ROOT / "media/virtual_bus_loop.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out.relative_to(ROOT)}")
    c.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="/tmp/so101_bus")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--rate", type=float, default=0.0, help="loop rate in Hz (0 = as fast as possible)")
    p.add_argument("--amplitude", type=int, default=300, help="wave amplitude [steps]")
    p.add_argument("--wave-period", type=float, default=3.0)
    p.add_argument("--timeout-ms", type=float, default=20.0)
    p.add_argument("--ramp-s", type=float, default=2.0, help="time to move to the ready pose")
    main(p.parse_args())
