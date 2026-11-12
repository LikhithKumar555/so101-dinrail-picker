#!/usr/bin/env python3
"""Send the arm through park -> ready -> above supply -> above rail -> park via ros2_control.

The joint targets were computed with our own IK from Phase 2 (scene frame = robot base).
Run while bringup.launch.py is running:   ros2 run so101_bringup demo_trajectory.py
"""
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# (name, joint angles [rad], time from start [s])
WAYPOINTS = [
    ("ready",        [-0.003, -0.198, 0.810, 0.959, 0.046], 3.0),
    ("above supply", [-0.392, -0.262, 0.706, 1.126, 1.228], 5.0),
    ("above rail",   [0.469, 0.343, -0.050, 1.278, 2.088], 8.0),
    ("park",         [0.0, -1.745, -1.658, 0.262, 0.0], 11.0),
]


class Demo(Node):
    def __init__(self):
        super().__init__("so101_demo_trajectory")
        self.arm = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.gripper = self.create_publisher(Float64MultiArray, "/gripper_controller/commands", 10)

    def set_gripper(self, angle: float):
        self.gripper.publish(Float64MultiArray(data=[angle]))

    def run(self) -> bool:
        if not self.arm.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("arm_controller action server not available")
            return False
        self.set_gripper(0.35)  # open a little

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        for name, q, t in WAYPOINTS:
            p = JointTrajectoryPoint(positions=q, velocities=[0.0] * len(q))
            p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(p)
            self.get_logger().info(f"waypoint '{name}' at t = {t:.1f} s")

        send = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if not handle.accepted:
            self.get_logger().error("goal rejected")
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        code = res.result().result.error_code
        self.set_gripper(0.0)
        self.get_logger().info("done" if code == 0 else f"failed, error_code={code}")
        return code == 0


def main():
    rclpy.init()
    node = Demo()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
