#!/usr/bin/env python3
"""Simulated servo telemetry until the real arm arrives (Phase 5 replaces this node).

Physics, kept simple but honest:
  * load:        small idle holding load + noise (fraction of max torque)
  * temperature: first-order thermal model of the motor. Copper losses grow with current^2 and
                 current grows with torque, so the steady-state temperature rise ~ load^2:
                     dT/dt = (T_amb + K * load^2 - T) / tau
  * voltage:     12 V supply with a small drop under load

Fault injection (to test the safety monitor), e.g.:
  ros2 param set /fake_telemetry inject "overheat:elbow_flex"   # high load -> heats up slowly
  ros2 param set /fake_telemetry inject "overload:shoulder_lift" # blocked joint, load 0.95
  ros2 param set /fake_telemetry inject "undervoltage"
  ros2 param set /fake_telemetry inject "dropout"                # stop publishing
  ros2 param set /fake_telemetry inject ""                       # back to normal
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from so101_interfaces.msg import ServoTelemetry

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
T_AMB = 30.0        # degC, servo case temperature at no load
K_HEAT = 80.0       # degC steady-state rise at 100 % load
TAU = 60.0          # s, thermal time constant of a small servo motor
IDLE_LOAD = 0.15


class FakeTelemetry(Node):
    def __init__(self):
        super().__init__("fake_telemetry")
        self.rate = self.declare_parameter("rate_hz", 50.0).value
        self.declare_parameter("time_scale", 10.0)   # speed up heating for the demo
        self.declare_parameter("inject", "")
        self.pub = self.create_publisher(ServoTelemetry, "servo_telemetry", qos_profile_sensor_data)
        self.temp = np.full(len(JOINTS), T_AMB + K_HEAT * IDLE_LOAD**2)
        self.rng = np.random.default_rng(0)
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.last_inject = None

    def tick(self):
        inject = self.get_parameter("inject").value
        if inject != self.last_inject:
            self.get_logger().info(f"inject = '{inject}'")
            self.last_inject = inject
        if inject == "dropout":
            return

        load = IDLE_LOAD + 0.03 * self.rng.standard_normal(len(JOINTS))
        voltage = 12.0 - 0.3 * np.abs(load) + 0.02 * self.rng.standard_normal(len(JOINTS))
        kind, _, joint = inject.partition(":")
        if joint in JOINTS:
            i = JOINTS.index(joint)
            if kind == "overheat":
                load[i] = 0.75 + 0.02 * self.rng.standard_normal()   # high but below overload
            elif kind == "overload":
                load[i] = 0.95
        if kind == "undervoltage":
            voltage[:] = 5.2

        dt = self.get_parameter("time_scale").value / self.rate
        self.temp += (T_AMB + K_HEAT * load**2 - self.temp) * dt / TAU

        msg = ServoTelemetry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.load = load.tolist()
        msg.temperature = self.temp.tolist()
        msg.voltage = voltage.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(FakeTelemetry())


if __name__ == "__main__":
    main()
