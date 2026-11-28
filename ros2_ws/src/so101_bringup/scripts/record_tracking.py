#!/usr/bin/env python3
"""Record how well the arm follows a trajectory: reference vs. measured joint angles.

The JointTrajectoryController publishes on /arm_controller/controller_state, every cycle:
  reference  = where the interpolated trajectory says the joint should be now
  feedback   = what the hardware reports (state interfaces, i.e. the encoder)
  error      = reference - feedback
With mock hardware the error is always 0 (command = state). With the Feetech bus it shows the
real servo behaviour: P-control sag under gravity, velocity limits, quantisation, bus delay.

  ros2 run so101_bringup record_tracking.py --duration 14 --plot media/tracking.png
  (start it, then within a second: ros2 run so101_bringup demo_trajectory.py)
"""
import argparse
import csv
import time

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from so101_interfaces.msg import ServoTelemetry


class Recorder(Node):
    def __init__(self):
        super().__init__("record_tracking")
        self.t, self.ref, self.fb, self.joints = [], [], [], []
        self.load_t, self.load, self.load_names = [], [], []
        self.create_subscription(JointTrajectoryControllerState, "/arm_controller/controller_state",
                                 self.on_state, 100)
        self.create_subscription(ServoTelemetry, "/servo_telemetry", self.on_telemetry,
                                 qos_profile_sensor_data)

    def on_state(self, m):
        if not m.reference.positions or not m.feedback.positions:
            return
        self.joints = list(m.joint_names)
        self.t.append(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec)
        self.ref.append(m.reference.positions)
        self.fb.append(m.feedback.positions)

    def on_telemetry(self, m):
        self.load_names = list(m.name)
        self.load_t.append(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec)
        self.load.append(m.load)


def main(a):
    rclpy.init()
    node = Recorder()
    end = time.monotonic() + a.duration
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    if len(node.t) < 2:
        print("no controller_state received - is arm_controller active?")
        return 1
    t = np.array(node.t) - node.t[0]
    ref, fb = np.array(node.ref), np.array(node.fb)
    err = np.degrees(ref - fb)
    moving = np.abs(np.gradient(ref, t, axis=0)).max(axis=1) > 1e-3     # trajectory in motion
    print(f"{len(t)} samples over {t[-1]:.1f} s ({len(t) / max(t[-1], 1e-9):.0f} Hz)")
    print(f"{'joint':14s} {'rms err':>8s} {'max err':>8s} {'final err':>9s}   [deg]")
    for k, name in enumerate(node.joints):
        e = err[:, k]
        print(f"{name:14s} {np.sqrt(np.mean(e**2)):8.2f} {np.abs(e).max():8.2f} {e[-1]:9.2f}")
    if moving.any():
        print(f"while moving: rms error over all joints {np.sqrt(np.mean(err[moving]**2)):.2f} deg")
    if node.load:
        L = np.abs(np.array(node.load))
        print("max |load| " + "  ".join(f"{n}={v:.2f}" for n, v in zip(node.load_names, L.max(0))))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"ref_{n}" for n in node.joints] + [f"fb_{n}" for n in node.joints])
            for row in zip(t, ref, fb):
                w.writerow([f"{row[0]:.4f}"] + [f"{v:.5f}" for v in row[1]] + [f"{v:.5f}" for v in row[2]])
        print(f"wrote {a.csv}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(node.joints)
        fig, ax = plt.subplots(n, 2, figsize=(11, 1.9 * n), sharex=True)
        for k, name in enumerate(node.joints):
            ax[k, 0].plot(t, np.degrees(ref[:, k]), "k--", lw=1, label="reference")
            ax[k, 0].plot(t, np.degrees(fb[:, k]), lw=1.2, label="measured (encoder)")
            ax[k, 0].set_ylabel(f"{name}\n[deg]", fontsize=8)
            ax[k, 1].plot(t, err[:, k], lw=1)
            ax[k, 1].axhline(0, color="0.6", lw=0.5)
            ax[k, 1].set_ylabel("error [deg]", fontsize=8)
        ax[0, 0].legend(fontsize=7, loc="upper right")
        ax[0, 0].set_title("joint angle", fontsize=9)
        ax[0, 1].set_title("tracking error = reference - measured", fontsize=9)
        ax[-1, 0].set_xlabel("time [s]")
        ax[-1, 1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f"wrote {a.plot}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=14.0, help="seconds to record")
    p.add_argument("--csv", help="save reference and measured angles")
    p.add_argument("--plot", help="save a PNG")
    raise SystemExit(main(p.parse_args()))
