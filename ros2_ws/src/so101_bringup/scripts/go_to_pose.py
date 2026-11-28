#!/usr/bin/env python3
"""Move the arm to a named pose straight through the JointTrajectoryController (no planner).

  ros2 run so101_bringup go_to_pose.py park            # 4 s from wherever the arm is
  ros2 run so101_bringup go_to_pose.py ready --duration 6
  ros2 run so101_bringup go_to_pose.py --list

What it is for: after power-up the arm lies folded on its stops, wrist hanging down, gripper
resting ON the table. Planning from there is a bad idea, and a straight joint move is too: it
turns the wrist up first, which presses the gripper into the table. Friction holds it
(stick), the wrist servo saturates, then it breaks loose and jumps (slip) - the trajectory
controller sees > 0.2 rad path error and aborts. So, like a real robot "unfolding":

  1. lift-off: raise only shoulder_lift to LIFT_OFF (the gripper leaves the table), 2 s
  2. then move all joints to the target pose, --duration s

Step 1 is skipped when the shoulder is already above LIFT_OFF. The JointTrajectoryController
interpolates from where the joints ARE; it does not check limits or collisions on the way.
No collision checking: use it only for the stored, known-safe poses below (SRDF group_states).
"""
import argparse

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
POSES = {                                  # same values as the SRDF group_states
    "park":  [0.0, -1.745, -1.658, 0.262, 0.0],
    "ready": [-0.003, -0.198, 0.810, 0.959, 0.046],
}
LIFT_OFF = -1.2                            # shoulder_lift [rad] for step 1
LIMITS = {                                 # URDF joint limits [rad], for the report only
    "shoulder_pan": (-1.91986, 1.91986), "shoulder_lift": (-1.7500, 1.74533),
    "elbow_flex": (-1.69, 1.69), "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}


class GoTo(Node):
    def __init__(self):
        super().__init__("so101_go_to_pose")
        self.arm = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.q = None
        self.create_subscription(JointState, "/joint_states", self.on_state, 10)

    def on_state(self, m):
        pos = dict(zip(m.name, m.position))
        if all(j in pos for j in JOINTS):
            self.q = [pos[j] for j in JOINTS]

    def report(self, title):
        print(title)
        for j, v in zip(JOINTS, self.q):
            lo, hi = LIMITS[j]
            flag = "" if lo <= v <= hi else f"   <- OUTSIDE [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {j:14s} {v:+.4f} rad{flag}")

    def wait_for_state(self, timeout_s=5.0):
        end = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while self.q is None and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.q is not None

    def go(self, target, duration_s) -> bool:
        if not self.arm.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("arm_controller action server not available")
            return False
        waypoints = []                         # (positions, time from start [s])
        if self.q[1] < LIFT_OFF:
            lift = list(self.q)
            lift[1] = LIFT_OFF
            waypoints.append((lift, 2.0))
            print(f"step 1: lift-off (shoulder_lift -> {LIFT_OFF} rad), then step 2: '{target}'")
        t0 = waypoints[-1][1] if waypoints else 0.0
        waypoints.append((POSES[target], t0 + duration_s))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        for q, t in waypoints:
            p = JointTrajectoryPoint(positions=q, velocities=[0.0] * len(JOINTS))
            p.time_from_start = Duration(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
            goal.trajectory.points.append(p)
        send = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        if not send.result().accepted:
            self.get_logger().error("goal rejected")
            return False
        res = send.result().get_result_async()
        rclpy.spin_until_future_complete(self, res)
        return res.result().result.error_code == 0


def main(a):
    if a.list or a.pose is None:
        for name, q in POSES.items():
            print(f"{name:6s} {q}")
        return 0
    rclpy.init()
    node = GoTo()
    try:
        if not node.wait_for_state():
            print("no /joint_states - is bringup running?")
            return 1
        node.report("before:")
        ok = node.go(a.pose, a.duration)
        for _ in range(10):                    # let a few fresh joint_states arrive
            rclpy.spin_once(node, timeout_sec=0.05)
        node.report("after:")
        print(f"'{a.pose}' " + ("reached" if ok else "FAILED (see controller log)"))
        return 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", nargs="?", choices=sorted(POSES))
    ap.add_argument("--duration", type=float, default=4.0, help="seconds for the move")
    ap.add_argument("--list", action="store_true")
    raise SystemExit(main(ap.parse_args()))
