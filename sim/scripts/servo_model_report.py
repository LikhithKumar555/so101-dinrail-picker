"""Check the STS3215 model against its datasheet and show what the physics means for the arm.

  python sim/scripts/servo_model_report.py          -> numbers + media/servo_model.png

Panels: torque-speed and torque-current curves with the datasheet points, step response with
the default gain (P = 32) and LeRobot's softer gain (P = 16) under a gravity load, and heating
at stall until the servo's own overheat protection switches the torque off.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from so101_sim.feetech import protocol as P
from so101_sim.feetech.emulator import FeetechBusEmulator
from so101_sim.feetech.servo_model import KGCM, RAD_PER_STEP, ServoArray, ServoParams

ROOT = Path(__file__).resolve().parents[2]


def steady_speed(p, tau_load):
    s = ServoArray(1, p)
    s.torque_enable[:] = True
    s.goal_steps[:] = 4000                       # far ahead: full duty, the load opposes the motion
    s.step(tau_load=-abs(tau_load), n_steps=600)
    return abs(s.omega[0]), abs(s.current[0])


def main():
    p = ServoParams.sts3215_7v4()
    v = 7.4
    print("identified from the datasheet (7.4 V: 19.5 kg*cm stall, 52 rpm, 8 kg*cm/A, 5 kg*cm at 650 mA):")
    print(f"  k_t = {p.k_t:.3f} N*m/A   k_e = {p.k_e:.3f} V*s/rad   R = {p.r_25:.2f} ohm   "
          f"gearbox efficiency = {p.efficiency:.0%}")
    print(f"  friction at the output: dry {p.tau_coulomb * 1e3:.1f} mN*m + viscous {p.b_viscous * 1e3:.1f} mN*m*s/rad")
    print(f"  stall current = {v / p.r_25:.2f} A   reflected rotor inertia = {p.j_reflected * 1e3:.1f} g*m^2 "
          f"(rotor x 345^2)")
    tau_m = (p.j_reflected + 0.002) * p.r_25 / (p.k_t * p.k_e)
    print(f"  mechanical time constant with a 2 g*m^2 link: {tau_m * 1e3:.0f} ms")
    for kp in (32, 16):
        s = ServoArray(1, p)
        s.kp[:] = kp
        for load in (0.3, 0.6):
            e = s.steady_state_error_steps(load)
            print(f"  P = {kp:2d}: holding error at {load} N*m = {e:4.1f} steps = {np.degrees(e * RAD_PER_STEP):.2f} deg")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    loads = np.linspace(0, 1.85, 38)
    sw = np.array([steady_speed(p, t) for t in loads])
    a = ax[0, 0]
    a.plot(loads, sw[:, 0] * 60 / (2 * np.pi), label="model")
    a.plot([0, 19.5 * KGCM], [52, 0], "o", label="datasheet: no-load speed, stall torque")
    a.set_xlabel("load torque [N*m]"), a.set_ylabel("speed [rpm]"), a.set_title("torque - speed at 7.4 V")
    a.legend(), a.grid(alpha=0.3)
    a = ax[0, 1]
    a.plot(loads, sw[:, 1], label="model")
    a.plot([5 * KGCM], [0.65], "o", label="datasheet: rated 5 kg*cm at 650 mA")
    a.set_xlabel("load torque [N*m]"), a.set_ylabel("current [A]"), a.set_title("torque - current")
    a.legend(), a.grid(alpha=0.3)

    a = ax[1, 0]
    for kp in (32, 16):
        s = ServoArray(1, p, j_load=0.002)
        s.torque_enable[:] = True
        s.kp[:] = kp
        s.step(tau_load=-0.3, n_steps=2000)                    # settle under gravity first
        s.goal_steps[:] = 2048 + 200
        t, x = [], []
        for k in range(1200):
            s.step(tau_load=-0.3)
            t.append(k * s.dt * 1e3)
            x.append(s.position_steps[0])
        a.plot(t, x, label=f"P = {kp}")
    a.axhline(2248, color="k", ls=":", lw=1)
    a.set_xlabel("time [ms]"), a.set_ylabel("position [steps]")
    a.set_title("200-step move, 0.3 N*m gravity load, 2 g*m^2 link"), a.legend(), a.grid(alpha=0.3)

    a = ax[1, 1]
    bus = FeetechBusEmulator(ids=(1,))
    bus.handle_bytes(P.write(1, "Overload_Torque", 100))         # disable overload, show heating only
    bus.handle_bytes(P.write(1, "Torque_Enable", 1))
    bus.handle_bytes(P.write(1, "Goal_Position", 2548))
    tt, temp, tq = [], [], []
    for k in range(240):
        bus.advance(0.1, tau_load=-1.8)
        tt.append(bus.time)
        temp.append(bus.register(1, "Present_Temperature"))
        tq.append(bus.register(1, "Torque_Enable"))
    a.plot(tt, temp, label="temperature")
    a.axhline(70, color="r", ls=":", label="Max_Temperature_Limit")
    off = np.argmax(np.array(tq) == 0) if 0 in tq else None
    if off:
        a.axvline(tt[off], color="k", ls="--", label=f"torque off at {tt[off]:.1f} s")
    a.set_xlabel("time [s]"), a.set_ylabel("[degC]"), a.set_title("holding 1.8 N*m (near stall)")
    a.legend(), a.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "media/servo_model.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
