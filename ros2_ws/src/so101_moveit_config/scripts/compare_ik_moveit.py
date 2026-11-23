#!/usr/bin/env python3
"""MoveIt 2 half of the IK benchmark: solve the poses from experiments/ik_benchmark/poses.json
with move_group's /compute_ik service (pick_ik), and save the joint solutions.

  ros2 run so101_moveit_config compare_ik_moveit.py --repo ~/projects/so101-dinrail-picker
(needs move_group running: ros2 launch so101_moveit_config demo.launch.py)
"""
import argparse
import json
import time
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "projects" / "so101-dinrail-picker"))
    ap.add_argument("--timeout", type=float, default=0.1, help="s per IK request")
    a = ap.parse_args()
    folder = Path(a.repo).expanduser() / "experiments" / "ik_benchmark"
    data = json.loads((folder / "poses.json").read_text())

    rclpy.init()
    node = Node("compare_ik_moveit")
    cli = node.create_client(GetPositionIK, "/compute_ik")
    if not cli.wait_for_service(timeout_sec=10.0):
        raise SystemExit("/compute_ik not available - is move_group running?")
    seed = RobotState(joint_state=JointState(name=ARM + ["gripper"], position=data["seed_q"] + [0.0]))
    results = []
    for i, p in enumerate(data["poses"]):
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p["position"]
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = p["quat_xyzw"]
        req = GetPositionIK.Request(ik_request=PositionIKRequest(
            group_name="arm", robot_state=seed, avoid_collisions=False, pose_stamped=ps,
            ik_link_name="tcp_link", timeout=Duration(sec=0, nanosec=int(a.timeout * 1e9))))
        t0 = time.perf_counter()
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut)
        dt = (time.perf_counter() - t0) * 1000
        res = fut.result()
        q = None
        if res.error_code.val == MoveItErrorCodes.SUCCESS:
            js = res.solution.joint_state
            q = [js.position[js.name.index(j)] for j in ARM]
        results.append({"q": q, "time_ms": dt, "error_code": res.error_code.val})
        if (i + 1) % 50 == 0:
            node.get_logger().info(f"{i + 1}/{len(data['poses'])}")
    (folder / "moveit_ik.json").write_text(json.dumps(results))
    n_ok = sum(r["q"] is not None for r in results)
    print(f"MoveIt returned a solution for {n_ok}/{len(results)} poses -> {folder / 'moveit_ik.json'}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
